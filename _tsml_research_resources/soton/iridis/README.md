serial_scripts:
    Simple(er) scripts which submit each experiment as a separate job. This is fine
    for small numbers of experiments, but limits you to 32 single-threaded jobs and
    redirects to the serial queue when submitting to the batch queue on Iridis 5.

batch_scripts:
    More complex scripts which submit multiple experiments in a single job. This
    allows a much larger number of jobs for experiments when using batch queues,
    but requires more setup to work efficiently.

thread_scripts:
    Similar to serial_scripts but for multi-threaded jobs.

gpu_scripts:
    Scripts for running GPU jobs. These are similar to the serial scripts, but
    use the GPU queue and request GPUs.
