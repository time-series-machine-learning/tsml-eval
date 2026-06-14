serial_scripts:
    Simple(er) scripts which submit each experiment as a separate job and requests 1 core. 
    This is fine for small numbers of experiments.

batch_scripts:
    More complex scripts which submit multiple experiments and core requests in a 
    single job. This allows a much larger number of jobs submitted at once, but requires 
    more setup to work efficiently and can waste resources if there is a large difference
    in completion time for experiments.

thread_scripts:
    Similar to serial_scripts and batch_scripts but for multithreaded jobs requesting
    multiple cores per experiment.

gpu_scripts:
    Scripts for running GPU jobs. These are similar to the serial scripts, but
    use the GPU queues and request GPUs.
