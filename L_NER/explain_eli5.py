import eli5
import joblib

crf = joblib.load('L_NER_CRF_model_106_MERGED.pl')

setattr(crf, 'keep_tempfiles', False)
setattr(crf, 'model_filename', 'model.pl')

print(crf, type(crf))
#exit()

#eli5_obj = eli5.sklearn_crfsuite.explain_weights_sklearn_crfsuite(crf=crf,top=100)
#eli5_obj = eli5.show_weights(estimator=crf,top=100)
#eli5.show_weights(estimator=crf,top=100)
#eli5_obj = eli5.show_weights(estimator=crf,top=100,feature_re=None)
eli5_obj = eli5.show_weights(estimator=crf,top=20,feature_re=None)
#eli5_obj = eli5.show_weights(estimator=crf,top=100,feature_re='.+')
#eli5_obj = eli5.explain_weights(crf,top=100)

print(dir(eli5_obj))

html = eli5_obj._repr_html_()

f = open('out.html', 'wt', encoding='utf-8')
f.write(html)
f.close()
