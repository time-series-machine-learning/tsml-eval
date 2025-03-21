serial_scripts:
    Simple(er) scripts which submit each experiment as a separate job. This is fine
    for small numbers of experiments, but limits you to 32 single-threaded jobs and
    redirects to the serial queue when submitting to batch on Iridis 5.
batch_scripts:
    More complex scripts which submit multiple experiments in a single job. This
    allows a much larger number of jobs for experiments and allows using the
    batch queues, but requires more setup to work efficiently.
thread_scripts:
    Similar to serial_scripts but for multi-threaded jobs.
```
