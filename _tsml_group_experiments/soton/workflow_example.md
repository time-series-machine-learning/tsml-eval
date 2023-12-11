## Connecting to Iridis 5

Log in servers

iridis5_a.soton.ac.uk
iridis5_b.soton.ac.uk
iridis5_c.soton.ac.uk
iridis5_d.soton.ac.uk

dont use iridis5_a.soton.ac.uk because apparently everyone does. Need to be on the VPN

connect
ssh userid@iridis5_a.soton.ac.uk

ajb2u23@iridis5_a.soton.ac.uk

I am ajb2u23, matthew is mbm1g23

there is a Teams group for HPC called "HPC Community". log on, can then use github etc.
To create the environment, first do this once only:

https://sotonac.sharepoint.com/teams/HPCCommunityWiki/SitePages/Getting-Started-with-HPC.aspx

there is a Teams group for HPC called "HPC Community"

https://sotonac.sharepoint.com/teams/HPCCommunityWiki/SitePages/Slurm.aspx



## Installing tsml-eval

will need to do this once only

>conda init bash

then it all works for install.
> git clone https://github.com/time-series-machine-learning/tsml-eval
etc

The differences in script are this

>#SBATCH --nodes=1

and max run time is 60 hours.

Queue names:
https://sotonac.sharepoint.com/teams/HPCCommunityWiki/SitePages/Iridis%205%20Job-submission-and-Limits-Quotas.aspx

Note the queuing doesnt seem to work, and you run into this message

>sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job
submit limit, user's size and/or time limits)

need to fix this
