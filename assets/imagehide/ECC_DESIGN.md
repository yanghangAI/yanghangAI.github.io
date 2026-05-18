# Payload, ECC, and the Slepian-Wolf pHash trick

How we get a 1024-bit channel codeword to carry a 128-bit cryptographic
binding to the cover image, a 512-bit signature, and a 256-bit public key —
**when the literal bit budget is short** — by exploiting the fact that the
receiver has a noisy copy of one of those fields for free.

## What the payload needs to do

We want each watermarked image to carry a payload that proves provenance:

- **`H`** (128 bits): a perceptual hash (PDQ) of the original cover image.
  Binds the signature to the visual content — if someone modifies the
  image substantially, H changes and the signature stops verifying.
- **`sig`** (512 bits): Ed25519 detached signature over `H`.
- **`pk`** (256 bits): the public key the receiver verifies against.

Logical payload size: **128 + 512 + 256 = 896 bits**.

The model's channel codeword is **1024 bits** wide, so we have **128
"extra" bits** to spend on channel ECC. That sounds like enough for
RS(128, 112) — corrects 8 byte errors in 128 bytes. Marginal.

## The naive scheme (and why it's not enough)

Just pack everything and add some RS parity:

```
[ H | sig | pk | RS_parity ]
  128b 512b 256b  128b        = 1024 bits, fits the codeword
```

RS(128, 112) over GF(2^8): corrects 8 byte errors. For our typical
channel attacks (JPEG q=50, social-platform chains), the demo measures
~10-15 byte errors after the model decoder threshold — sometimes inside
budget, often not. Too tight.

We need **more parity** without dropping any of the cryptographic fields.
The only field that's not strictly required end-to-end is H — *because
the receiver can re-derive H from the (possibly attacked) image itself*.

## The Slepian-Wolf insight

[Slepian-Wolf coding](https://en.wikipedia.org/wiki/Slepian%E2%80%93Wolf_coding)
(1973) says: if the receiver has *side information* that's statistically
close to the source, you can transmit only the **conditional entropy** —
not the full source.

In our setting:
- **Source**: H (the original pHash, 128 bits).
- **Side info at receiver**: H' (pHash recomputed from the attacked
  image), which is *close* to H but not identical.
- The attack only flips a few bits — empirically PDQ drifts ≤ 3 bits
  on our 30-image × 11-attack sweep.

So the receiver "almost knows" H. We just need to transmit enough to
help it correct the few bits where it's wrong.

## BCH syndrome as the side-channel

[BCH(127, k, t)](https://en.wikipedia.org/wiki/BCH_code) is a binary
linear block code over GF(2^7) that corrects up to **t bit flips**. The
syndrome of a 127-bit vector is a 7t-bit summary; two vectors that
differ in ≤ t bit positions can be distinguished by their syndromes.

Our scheme: sender transmits **`S(H)` — the BCH syndrome of H** —
instead of H itself. Receiver does:

```
S(H')  = compute syndrome of locally-derived pHash      # free
S(H)   = received bits                                  # from channel
S(e)   = S(H) ⊕ S(H')                                   # syndrome of the error vector
e      = BCH_decode(S(e))                               # find ≤ t bit positions to flip
H      = H' ⊕ e                                         # recovered original H
```

This works as long as **the channel didn't corrupt `S(H)` too much** and
**the actual pHash drift ≤ t bits**.

### Choosing t

PDQ's measured max drift on our test bench: **3 bits** out of 128 across
all attack conditions. We use **BCH(127, t=4)** — one bit of margin.

- Code: BCH(127, 99, t=4)
- Syndrome length: 7 × 4 = **28 bits**
- Bits saved vs transmitting the full H: 128 − 28 = **100 bits**

(PDQ truncates the 256-bit standard hash to 128 bits via the zigzag of
the low-frequency 16×16 sub-block. BCH(127) operates on 127 bits, so we
drop the last bit of H — its LSB is also cleared sender-side in
`canonicalizeH()` so the BCH input matches.)

## The wire payload

Replace H with its 28-bit syndrome:

```
[ S(H) | sig | pk ]
   28b  512b 256b   = 796 bits  ← "wire payload"
```

The saved 100 bits get reallocated entirely to **RS outer parity**.

## RS(128, 100) outer code

The wire payload is **796 bits**, pad with 4 zero bits to reach
**100 bytes** = **K**, then RS-encode to **N = 128 bytes = 1024 bits**.

```
[ wire | 0000 ]  →  RS(128, 100) over GF(2^8)  →  [ 128 codeword bytes ]
   796b   4b           prim poly 0x11D                 = 1024 bits
                       generator g(x) = ∏(x - α^i), i ∈ [0, 28)
```

RS corrects up to **14 byte errors** anywhere in the 1024-bit codeword
(NSYM = 28, t = NSYM/2 = 14). That's almost 2× the 8 we had in the
naive scheme.

Implementation: `assets/imagehide/ecc.js`.

## Putting it together — encode

```
H          = canonicalizeH(PDQ(cover))                  # 128 bits, LSB cleared
S          = bchEncodeSyndrome(H[0..127))               # 28 bits
sig        = Ed25519.sign(H, sk)                        # 512 bits
pk         = sk.publicKey                               # 256 bits
wireBits   = pack(S, sig, pk)                           # 796 bits
codeword   = eccEncode(wireBits)                        # 1024 bits  ← model input
container  = model.embed(cover, codeword)
```

## Putting it together — decode

```
recCodeword = model.extract(attacked_container)         # 1024 bits, noisy
{ wireBits, eccErrors, eccOk } = eccDecode(recCodeword) # RS — fixes ≤ 14 byte errors
{ S_rx, sig, pk } = unpack(wireBits)

H_local    = canonicalizeH(PDQ(attacked_container))     # noisy side info
{ H, bchErrors, bchOk } = bchDecode(S_rx, H_local)      # XOR syndromes, find ≤ 4 bit flips
verified   = Ed25519.verify(sig, H, pk)
```

## Error budgets summary

| layer | code | corrects |
|---|---|---|
| Outer (channel) | RS(128, 100) over GF(2^8) | 14 byte errors anywhere in 1024-bit codeword |
| Inner (pHash) | BCH(127, t=4) Slepian-Wolf | 4 bit flips of pHash drift (PDQ observed max: 3) |

If RS fails (>14 byte errors), the wire payload is corrupted; in
particular S_rx may be wrong and BCH falls over downstream.
If BCH fails (>4 bits of pHash drift), the signature verification fails
even though the wire payload was recovered cleanly — the recovered H is
wrong by at least one bit.

## Why this is cheaper than just shipping H

Bit count comparison (logical 128+512+256 = 896 cryptographic bits):

| scheme | wire size | RS parity | RS corrects | pHash margin |
|---|---|---|---|---|
| Naive (transmit full H) | 896 b | 128 b | 8 byte errors | n/a |
| Slepian-Wolf (transmit S(H)) | 796 b | 228 b | 14 byte errors | 4 bits |

We're 2 bits short of doubling our channel error budget, with the same
1024-bit codeword. The bandwidth we recovered came from the receiver's
copy of the cover image — Shannon's free side information.

## Custom-text mode (no signature)

When the user types a message instead of using auto mode, the 96 bytes
of `sig` + `pk` slots carry their UTF-8 text. The H slot still holds
the real pHash so the BCH/Slepian-Wolf trick continues to work. The
RS error budget is unchanged; the message round-trips losslessly when
RS succeeds.

Implementation: `app.js → runEncode/runDecode`, `attacks.js → parseCustomText`.

## Where the math is in the code

- **PDQ** → `pdq.js` (Jarosz blur + 64×64 DCT + 16×16 zigzag → 128 bits)
- **BCH(127, t=4)** → `bch.js` (`bchEncodeSyndrome`, `bchDecode`)
- **RS(128, 100)** → `ecc.js` (`eccEncode`, `eccDecode`)
- **Glue** → `payload.js` (`packPayload`, `unpackPayload`, bit/byte conversions)
- **Encode flow** → `app.js → runEncode`
- **Decode flow** → `app.js → runDecode` (RS first, then unpack, then BCH)

## Why BCH and not RS for the inner code

pHash drift is **scattered random bit flips** — typically 1-3 isolated
positions across 128 bits. RS over GF(2^8) treats each byte as a single
unit; one corrupted bit flips an entire "byte error." Wasteful for
single-bit drift.

Binary BCH over GF(2^7) treats each bit independently. For a fixed
syndrome budget, BCH corrects ~16× more bit flips than RS would on the
same channel.
