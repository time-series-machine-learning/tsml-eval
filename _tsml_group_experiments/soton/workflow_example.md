Log in servers

ridis5_a.soton.ac.uk
iridis5_b.soton.ac.uk
iridis5_c.soton.ac.uk
iridis5_d.soton.ac.uk

dont use a because apparently everyone does. Need to be on the VPN

connect
ssh userid@iridis5_a.soton.ac.uk

ajb2u23@iridis5_a.soton.ac.uk

I am ajb2u23, matthew is mbm1g23

there is a Teams group for HPC called "HPC Community". log on, can then use github etc.
To create the environment, first do this once only:

conda init bash

then it all works for install. To run the script in soton directory, you might need
to run

module unload python

The script needs
this

#SBATCH --nodes=1

and max run time is 60 hours.

instead of adding python it adds

module add conda
