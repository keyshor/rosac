#!/usr/bin/sh

ARGS=$(getopt -o 'v:p:r:' --long 'gpu:,path:,run:' -- "$@") || exit
eval "set -- $ARGS"

gpu=3
run=1
path=experiment_models
progfile=progress$run.txt
logfile=log$run.txt

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
		(-r|--run)
			run=$2
			shift 2
			;;
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

echo starting run $run >> $progfile
echo basic: python cegrl.py -d $path/basic -g -v $gpu -n $run >> $progfile
python cegrl.py -d $path/basic -g -v $gpu -n $run >> $logfile
echo dagger: python cegrl.py -d $path/dagger -g -v $gpu -n $run -c >> $progfile
python cegrl.py -d $path/dagger -g -v $gpu -n $run -c >> $logfile
echo dynamic reward: python cegrl.py -d $path/svm -g -v $gpu -n $run -c -z >> $progfile
python cegrl.py -d $path/svm -g -v $gpu -n $run -c -z >> $logfile
echo ensemble: python cegrl.py -d $path/ensemble -g -v $gpu -n $run -c -z -e 3 >> $progfile
python cegrl.py -d $path/ensemble -g -v $gpu -n $run -c -z -e 3 >> $logfile
echo masac: python masac.py -d $path/masac -g -v $gpu -n $run -z >> $progfile
python masac.py -d $path/masac -g -v $gpu -n $run -z >> $logfile
echo ending run $run >> $progfile
