python3 opt.py facebook/opt-6.7b wikitext2 --sparsity 0.5 --wbits 5 --save /parsadata1/lisa/experiments/q_then_s/6.7b/hbfp6
python3 opt.py facebook/opt-6.7b wikitext2 --sparsity 0.5 --prunen 2 --prunem 4 --wbits 5 --save /parsadata1/lisa/experiments/q_then_s/6.7b/hbfp6-str
python3 opt.py facebook/opt-6.7b wikitext2 --sparsity 0.5 --prunen 2 --prunem 4 --wbits 7 --save /parsadata1/lisa/experiments/q_then_s/6.7b/hbfp8-str