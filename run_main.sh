for i in {5..200..5}
do
    check="checkpoint_$(printf "%03d" "${i}").pt"
    echo ${check}
    python main.py --checkpoint ${check}
done
