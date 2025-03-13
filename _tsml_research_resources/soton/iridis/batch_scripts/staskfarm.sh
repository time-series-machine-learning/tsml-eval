#!/bin/bash

# 2014-07-07 <paddy@tchpc.tcd.ie>
# Simple taskfarm script for a Slurm environment.
#
# Purpose: take a file of tasks (one per line) and create slurm multi-prog
# config to execute those tasks. Each task can comprise of multiple commands.
#
# Background: the slurm multi-prog setup can be difficult for some
# scenarios:
# * only one executable can be specified per task (e.g. no chain of commands
#   or shell loops are possible, such as "cd dir01; ./my_exec")
# * a limitation on the maximum number of characters per task description (256)
# * building the multi-prog file can be onerous, if you do not have the
#   luxury of using the '%t' tokens in your commands or arguments
# * the number of commands must match exactly the number of slurm tasks (-n),
#   which means updating two files if you wish to add or remove tasks
#
# 2017-10-17 <paddy@tchpc.tcd.ie>
# Inspired by forked version by cmeesters, add a second mode of operation:
# * original: staskfarm commands.txt
# * new: staskfarm command param [param]...
#   In this mode, allow a parameter sweep, prefaced by a single command. It
#   may be easier than creating the equivalent commands.txt, and allows
#   for shell globbing for example to generate the param list.


function usage {
	cat <<-EOM
	Usage: $(basename "$0") [-v] command_filename
	  or:  $(basename "$0") [-v] command param [param]...
	
	In the first mode of operation: $(basename "$0") [-v] command_filename
	
	The <command_filename> must have one individual task per
	line. The task can comprise of multiple bash shell commands,
	each separated by a semi-colon (;).
	
	For example, the following shows 6 tasks:
	
	    ./my_prog my_input01 > my_output01
	    ./my_prog my_input02 > my_output02
	    ./my_prog my_input03 > my_output03
	    ./my_prog my_input04 > my_output04
	    ./my_prog my_input05 > my_output05
	    ./my_prog my_input06 > my_output06
	
	A more complex example, showing 4 tasks which include loops:
	
	    cd sample01; for i in controls patients; do ./my_prog \$i; done
	    cd sample02; for i in controls patients; do ./my_prog \$i; done
	    cd sample03; for i in controls patients; do ./my_prog \$i; done
	    cd sample04; for i in controls patients; do ./my_prog \$i; done
	
	Enabling verbose mode prints each command to stdout as it is
	read from the command file.
	
	In the second mode of operation: $(basename "$0") [-v] command param [param]...
	
	The <command> is combined with each of the individual <param> parameters to
	generate the list of tasks to be executed. The number of tasks will be equal
	to the number of <param> values.
	
	The <param> values can either be a simple list (e.g. input1 input2...),
	or a shell glob (e.g. *.inp).
	
	Note that no output redirection is performed in this mode.
	
	Limitations:

  * the use of MPI is not supported in the tasks. Only serial tasks
    can appear in the task lists.
	
	* it writes the list of tasks to K files, where K is the value of
	  of the SLURM_NTASKS environment variable. The tasks are written
	  in a simple round-robin manner over the K files. This makes no
	  provision for how quickly any individual task might execute
	  compared to the others, and so an equal division of labour
	  between the SLURM_NTASKS processors is not guaranteed at all.
	
	* it makes no decisions about memory usage per task. The
	  assumption is that the user has already calculated memory
	  consumption, and has used a combination of "#SBATCH -n <n>"
	  and "#SBATCH -N <N>" to fit. For example, if the node has 8
	  cores and 16 GB of RAM, then "#SBATCH -n 8" will spread the
	  tasks over 8 cores on one machine, and will assume that the
	  total memory usage is no more than 16GB (2GB per task). If you
	  need 4GB per task, then instead you must use "#SBATCH -n 8"
	  and "#SBATCH -N 2" in order to spread the 8 tasks
	  over 2 nodes.
	
	* no output redirection is performed, so any stdout/stderr will
	  be sent to the slurm-NNNNN.out file by default. This can
	  be changed by adding individual redirects to each task (in the
	  first mode of operation). Care must be taken in that case so
	  that the output files have unique names/paths.
	
	Note that this program will create a temporary directory
	(called .taskfarm_job_\${SLURM_JOB_ID}) in which to store
	the slurm multi-config files.
	
EOM
}


######################################################
# Variables
######################################################
verbose=0
command_filename=""
# no-op dummy command, for when ncommands < SLURM_NTASKS
dummy_command=/bin/true
# loop counter
i=0
# count of non-blank and non-comment lines: the number of actual commands
ncommands=0


######################################################
# Parse options
######################################################
while getopts "vh" Option
do
	case "$Option" in
		v) verbose="1";;
		h) usage; exit;;
		*) usage; exit;;
	esac
done
shift $((OPTIND-1))


######################################################
# Check for command_filename
######################################################
if [ "$#" -eq "0" ]
then
	usage; exit 1
fi


######################################################
# Check for slurm environment
######################################################
if [ "x${SLURM_JOB_ID}" = "x" -o "x${SLURM_NTASKS}" = "x" ]
then
	echo "$(basename "$0"): error: must be executed from within a SLURM allocation. Exiting."
	exit 1
fi


######################################################
# Sanity check if old stale files exist
######################################################
if [ -d ".taskfarm_job_${SLURM_JOB_ID}" ]
then
	if [ "${verbose}" = "1" ]
	then
		echo "Deleting old job files .taskfarm_job_${SLURM_JOB_ID}/*.sh"
	fi
	rm -f .taskfarm_job_"${SLURM_JOB_ID}"/*.sh
fi


######################################################
# Create the taskfarm directory
######################################################
if [ "${verbose}" = "1" ]
then
	echo "Creating taskfarm job directory .taskfarm_job_${SLURM_JOB_ID}"
fi
mkdir ".taskfarm_job_${SLURM_JOB_ID}"


######################################################
# Two modes of operation:
# 1 if ($# == 1) then we've supplied a command_filename
#   and go with the previous logic
# 2 else we've supplied a command and a list of params,
#   so generate the tasks based on those
######################################################


if [ "$#" -eq "1" ]
then
	######################################################
	# Mode 1: command_filename
	######################################################

	command_filename=$1

	######################################################
	# Does the file exist?
	######################################################
	if [ ! -f "${command_filename}" ]
	then
		echo "$(basename "$0"): error: commands file ${command_filename} does not exist. Exiting."
		exit 1
	fi


	if [ "${verbose}" = "1" ]
	then
		echo ""
		echo "-------------------- $(basename "$0") START --------------------"
		echo "Reading commands from file: ${command_filename}."
		echo "There are $(wc -l < "${command_filename}") lines in the file."
		echo "They will be spread over the ${SLURM_NTASKS} processors:   ${SLURM_TASKS_PER_NODE} tasks on ${SLURM_NODELIST}"
		#md5sum ${command_filename}
	fi


	######################################################
	# Warn if no output redirection
	######################################################
	if [ "${verbose}" = "1" ] && ! grep -q '>' "${command_filename}"
	then
		echo ""
		echo "WARNING: there is no individual task output redirection in the ${command_filename}"
		echo "         file. This could potentially be a problem. Output of all individual"
		echo "         tasks will likely be merged in the slurm output file ('slurm-${SLURM_JOB_ID}.out')."
		echo ""
	fi


	######################################################
	# Main loop:
	# read the file, line by line, and create the
	# individual multi-prog shell scripts
	######################################################
	while read line
	do
		# ignore blank lines and comment lines
		if [[ $line =~ ^[[:blank:]]*# || $line =~ ^[[:blank:]]*$ ]]
		then
			if [ "${verbose}" = "1" ]
			then
				echo "Skipping blank and comment lines"
			fi
			continue
		fi

		if [ "${verbose}" = "1" ]
		then
			echo "Adding the following line to .taskfarm_job_${SLURM_JOB_ID}/${i}.sh:     $line"
		fi

		echo "$line" >> ".taskfarm_job_${SLURM_JOB_ID}/${i}.sh"

		# increment, modulo the number of tasks
		(( i = (i + 1) % SLURM_NTASKS ))

		# increment the total number of commands
		(( ncommands++ ))
	done < "${command_filename}"

else
	######################################################
	# Mode 2: command + param list
	######################################################

	command=$1

	# the first positional arg is the 'command' variable, which we've saved above.
	# remove it from the list
	shift

	######################################################
	# Does the file exist?
	######################################################
	if [ ! "which ${command}" ]
	then
		echo "$(basename "$0"): error: taskfarm command ${command} does not exist. Exiting."
		exit 1
	fi


	if [ "${verbose}" = "1" ]
	then
		echo ""
		echo "-------------------- $(basename "$0") START --------------------"
		echo "Using command: ${command} with $# parameters."
		echo "They will be spread over the ${SLURM_NTASKS} processors:   ${SLURM_TASKS_PER_NODE} tasks on ${SLURM_NODELIST}"
		#md5sum ${command_filename}
	fi


	######################################################
	# Warn if no output redirection
	######################################################
	if [ "${verbose}" = "1" ]
	then
		echo ""
		echo "WARNING: there is no individual task output redirection."
		echo "         This could potentially be a problem. Output of all individual"
		echo "         tasks will likely be merged in the slurm output file ('slurm-${SLURM_JOB_ID}.out')."
		echo ""
	fi


	######################################################
	# Main loop:
	# loop over the parameters, and create the
	# individual multi-prog shell scripts
	######################################################

	# just in case of quoted parameters (e.g. filenames with spaces)
	SAVEIFS=$IFS
	IFS=$(echo -en "\n\b")

	for param in "$@"
	do
		line="${command} ${param}"

		if [ "${verbose}" = "1" ]
		then
			echo "Adding the following line to .taskfarm_job_${SLURM_JOB_ID}/${i}.sh:     $line"
		fi

		echo "$line" >> ".taskfarm_job_${SLURM_JOB_ID}/${i}.sh"

		# increment, modulo the number of tasks
		(( i = (i + 1) % SLURM_NTASKS ))

		# increment the total number of commands
		(( ncommands++ ))
	done

	IFS=$SAVEIFS
fi

######################################################
# Sanity check: if the number of commands is less
# than SLURM_NTASKS, then it the srun --multi-prog
# will error. Add dummy tasks to fill them out.
######################################################

while (( ncommands < SLURM_NTASKS ))
do
	if [ "${verbose}" = "1" ]
	then
		echo "Adding the dummy command to .taskfarm_job_${SLURM_JOB_ID}/${i}.sh:     ${dummy_command}"
	fi

	echo "${dummy_command}" >> ".taskfarm_job_${SLURM_JOB_ID}/${i}.sh"

	# increment (don't need the modulo here, the loop guard takes care of it)
	(( i++ ))

	# increment the total number of commands
	(( ncommands++ ))
done


######################################################
# Create the multi-prog file (using the '%t' token)
######################################################
if [ "${verbose}" = "1" ]
then
	echo "Creating the .taskfarm_job_${SLURM_JOB_ID}/multi.config file."
fi
echo "* bash .taskfarm_job_${SLURM_JOB_ID}/%t.sh" > ".taskfarm_job_${SLURM_JOB_ID}/multi.config"


######################################################
# And finally run the slurm multi-prog task file
######################################################
if [ "${verbose}" = "1" ]
then
	echo "About to execute 'srun --multi-prog .taskfarm_job_${SLURM_JOB_ID}/multi.config'."
	echo "==================== $(basename "$0") END ===================="
	echo ""
fi
srun --multi-prog ".taskfarm_job_${SLURM_JOB_ID}/multi.config"

