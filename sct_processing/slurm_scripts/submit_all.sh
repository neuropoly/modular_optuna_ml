for f in *.sl; do
if [[ "$f" != "template.sl" ]]; then
    sbatch "$f"
fi
done