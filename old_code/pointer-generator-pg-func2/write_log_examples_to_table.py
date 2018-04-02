

examples_path = 'cnn_dailymail_236000_cov_examples'
examples_table_path = 'cnn_dailymail_236000_cov_examples_table'

f = open(examples_path)
lines = f.read().splitlines()
f.close()

art = 'INFO:tensorflow:ARTICLE:  '
ref = 'INFO:tensorflow:REFERENCE SUMMARY: '
gen = 'INFO:tensorflow:GENERATED SUMMARY: '

f = open(examples_table_path, 'w')

for l in lines:
    if art in l:
        l = l.replace(art, '')
        f.write(l + '\t')
    elif ref in l:
        l = l.replace(ref, '')
        f.write(l + '\t')
    elif gen in l:
        l = l.replace(gen, '')
        f.write(l + '\n')

f.close()
