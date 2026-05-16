# yanghangAI.github.io

Personal academic site for Hang Yang, PhD candidate in Mathematics at UMass Amherst. Built with Jekyll on GitHub Pages. The home page is a single-scrolling CV; `/tools/` lists personal projects (currently the live cluster dashboard at `/cluster/` and an external link to [What2Do](https://yanghangAI.github.io/what2do/)).

The cluster dashboard is fed by `scripts/update-cluster-dashboard.sh`, which runs via cron on the UMass Unity HPC cluster and pushes a refreshed `assets/cluster-data.json` every 15 minutes.
