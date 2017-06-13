#!/usr/local/bin/bash

input=$1                        # /home/shigashi/data_shigashi/resources/BCCWJ1.1/Disk4/TSV_SUW_OT/*/*.txt
tmp=${input##*/}
name=${tmp%%.*}

out_dir=/home/shigashi/data_shigashi/resources/BCCWJ1.1/Disk4/processed/ma2/
out_train=${out_dir}${name}_train.tsv
out_val=${out_dir}${name}_val.tsv
out_test=${out_dir}${name}_test.tsv
echo $input "->" $out_train

p1=9776                         # prob for class1
p2=112                          # prob for class2
p3=112                          # prob for class2
#echo $p1 $p2 $p3

d1=$p1
d2=$(($d1+$p2))
d3=$(($d2+$p3))
#echo $d1 $d2 $d3

N=`cut -f10 $input | grep B | wc -l | cut -d' ' -f1`
n2=$(($N*$p2/10000))
n3=$(($N*$p3/10000))
n1=$(($N-$n2-$n3))
echo $N $n1 $n2 $n3


# read file

: > $out_train
: > $out_val
: > $out_test

label='x'
cat $input | cut -f10,17,24 | while read line ; do
    IFS=''
    bos=`echo $line | cut -f1`
    pos=`echo $line | cut -f2`
    word=`echo $line | cut -f3`

    if [ $bos = 'B' ]; then
        while true; do
            # if [ $(($n1+$n2+$n3)) -le 0 ]; then

            r=$(($RANDOM))              # 0-32767
            if [ $r -ge 30000 ]; then   # 0-29999
                continue
            fi

            r=$(($r%10000+1))           # 1-10000
            if [ $n1 -gt 0 -a $r -le $d1 ]; then
                label='c1'
                n1=$(($n1-1))
            elif [ $n2 -gt 0 -a $r -le $d2 ]; then
                label='c2'
                n2=$(($n2-1))
            elif [ $n3 -gt 0 -a $r -le $d3 ]; then
                label='c3'
                n3=$(($n3-1))
            else
                continue
            fi

            break
        done
    fi

    # echo $label $r $n1 $n2 $n3
    if [ $label = 'c1' ]; then
        echo $bos$'\t'$word$'\t'$pos$ >> $out_train
    elif [ $label = 'c2' ]; then
        echo $bos$'\t'$word$'\t'$pos$ >> $out_val
    else
        echo $bos$'\t'$word$'\t'$pos$ >> $out_test
    fi        
done
