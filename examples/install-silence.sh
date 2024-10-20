source ~/.bashrc
ca dxtb-dev

cd ../
pip install . > pip.log
cd examples

python test.py