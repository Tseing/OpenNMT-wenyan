import thulac


sep_model = thulac.thulac(seg_only=True)
sep_model.cut_f('input.txt', 'input_sep.txt')
