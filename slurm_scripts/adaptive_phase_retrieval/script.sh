for i in `find ./15_07_small_teacher/quad -name "no*20.sh" -type f`; do
    sbatch $i
done
