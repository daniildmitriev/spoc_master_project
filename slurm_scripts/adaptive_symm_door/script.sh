for i in `find ./30_05_hidden -name "*.sh" -type f`; do
    sbatch $i
done
