source ~/.bashrc
cadxtb
export XTBPATH=/scratch/pawsey0799/yx7184/xtb_install/share/xtb
xtb $1 --opt --ohess --grad --gfn 2 --output $1.log