cd ../data/
git clone --depth=1 --branch=master https://github.com/ZhaofengWu/counterfactual-evaluation.git counterfactual-evaluation-original
rm -rf ./counterfactual-evaluation-original/.git

cd counterfactual-evaluation-original

cd ../preprocess
python counterfactual.py
python counterfactual_part2.py