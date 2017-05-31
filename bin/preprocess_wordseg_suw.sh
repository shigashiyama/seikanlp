#!/usr/local/bin/bash

in_dir='TSV_SUW_OT'
out_dir='SUW_wordseg'


for path in `find ${in_dir}/*/*`; do
    mid_dir=${path#*/}
    mid_dir=${mid_dir%/*}
    fn=`basename $path`
    fn2=${fn%.*}

    # mkdir -p ${out_dir}/${mid_dir}
    # echo ${in_dir}/${mid_dir}/${fn}
    # echo ${out_dir}/${mid_dir}/${fn2}-wseg.tsv

    out_file=${out_dir}/${fn2}-wseg.tsv
    : > $out_file

    echo $path

    bof=1
    cat ${in_dir}/${mid_dir}/${fn} | \
        awk -F$'\t' '{printf("%s\t%s\n", $10, $22)}' | \
        while read line; do
            bos=${line:0:1}
            word=${line:2}
            # echo "*" $bos $word
            
            if [ $bof = 0 -a $bos = 'B' ]; then
                echo '' >> $out_file
            fi
            
            for ((i=0; i<${#word}; i++)); do
                if [ $i = 0 ]; then
                    label='B'
                else
                    label='I'
                fi
                echo ${word:$i:1}$'\t'$label >> $out_file
            done
            bof=0
        done
done
