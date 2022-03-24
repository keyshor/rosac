#!/usr/bin/sh

ARGS=$(getopt -o 'v:a:p:f:l:' --long 'gpu:,algo:,path:,progfile:,logfile:' -- "$@") || exit
eval "set -- $ARGS"

gpu=3
algo=basic
flags=
path=cegrl_models
progfile=progress.txt
logfile=log.txt

while true
do
	case $1 in
		(-v|--gpu)
			gpu=$2
			shift 2
			;;
		(-a|--algo)
			algo=$2
			case $2 in
				(basic)
					flags=
					;;
				(dagger)
					flags='-c'
					;;
				(svm)
					flags='-f -z'
					;;
				(ensemble)
					flags='-f -z -e 3'
					;;
				(*)
					echo unrecognized algorithm $2
					exit 1
					;;
			esac
			shift 2
			;;
		(-p|--path)
			path=$2
			shift 2
			;;
		(-f|--progfile)
			progfile=$2
			shift 2
			;;
		(-l|--logfile)
			logfile=$2
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

for run in $(seq 3)
do
	echo starting run $run >> $progfile
	#echo python cegrl.py -d $path/$algo -g -v $gpu -n $run $flags
	#python cegrl.py -d $path/$algo -g -v $gpu -n $run $flags
	echo basic: python cegrl.py -d $path/basic -g -v $gpu -n $run >> $progfile
	python cegrl.py -d $path/basic -g -v $gpu -n $run >> $logfile
	echo dagger: python cegrl.py -d $path/dagger -g -v $gpu -n $run -c >> $progfile
	python cegrl.py -d $path/dagger -g -v $gpu -n $run -c >> $logfile
	echo svm: python cegrl.py -d $path/svm -g -v $gpu -n $run -c -z >> $progfile
	python cegrl.py -d $path/svm -g -v $gpu -n $run -c -z >> $logfile
	echo ensemble: python cegrl.py -d $path/ensemble -g -v $gpu -n $run -c -z -e 3 >> $progfile
	python cegrl.py -d $path/ensemble -g -v $gpu -n $run -c -z -e 3 >> $logfile
	echo ending run $run >> $progfile
done
