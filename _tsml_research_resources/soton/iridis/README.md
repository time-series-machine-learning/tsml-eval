serial_scripts:
    Simple(er) scripts which submit each experiment as a separate job. This is fine
    for small numbers of experiments, but limits you to 32 jobs per user on Iridis.
    
batch_scripts:
    More complex scripts which submit multiple experiments in a single job. This
    allows a much larger number of jobs for experiments, but requires more setup to
    work efficiently.
