#!/usr/bin/sh

ARGS=$(getopt -o 'v:p:r:' --long 'gpu:,path:,run:' -- "$@") || exit
eval "set -- $ARGS"

gpu=3
#run=1
path=experiment_models

while true
do
	case $1 in
		(-v|--gpu)
			gpu=$2
			shift 2
			;;
		(-p|--path)
			path=$2
			shift 2
			;;
		#(-r|--run)
		#	run=$2
		#	shift 2
		#	;;
		(--)
			shift
			break
			;;
		(*)
			echo unrecognized option $1
			exit 1
			;;
	esac
done

#progfile=progress$run.txt
#logfile=log$run.txt
progfile=progress.txt
logfile=log.txt

for run in $(seq 5)
do
	echo starting run $run >> $progfile
	echo NAIVE: python cegrl.py -d $path/NAIVE -g -v $gpu -n $run >> $progfile
	python cegrl.py -d $path/NAIVE -g -v $gpu -n $run >> $logfile
	echo DAGGER: python cegrl.py -d $path/DAGGER -g -v $gpu -n $run -c >> $progfile
	python cegrl.py -d $path/DAGGER -g -v $gpu -n $run -c >> $logfile
	echo AROSAC: python cegrl.py -d $path/AROSAC -g -v $gpu -n $run -c -z >> $progfile
	python cegrl.py -d $path/AROSAC -g -v $gpu -n $run -c -z >> $logfile
	#echo ensemble: python cegrl.py -d $path/ensemble -g -v $gpu -n $run -c -z -e 3 >> $progfile
	#python cegrl.py -d $path/ensemble -g -v $gpu -n $run -c -z -e 3 >> $logfile
	#echo masac: python masac.py -d $path/masac -g -v $gpu -n $run -z >> $progfile
	#python masac.py -d $path/masac -g -v $gpu -n $run -z >> $logfile
	echo ending run $run >> $progfile
done
