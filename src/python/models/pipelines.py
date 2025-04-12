from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import FeatureUnion

def get_pipeline_definitions():
    return {
        'vanilla': lambda model: Pipeline([
            ('scaler', 'passthrough'), 
            ('model', model)
        ]),
        # 'interact_select': lambda model: Pipeline([
        #     ('scaler', 'passthrough'), 
        #     ('interactions', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        #     ('select_features', SelectKBest(score_func=f_regression, k=100)),
        #     ('model', model)
        # ]),
        # 'pca_lda': lambda model: Pipeline([
        #     ('scaler', 'passthrough'), 
        #     ('feature_union', FeatureUnion([
        #         ('pca', PCA(n_components=0.95)),
        #         ('lda', LinearDiscriminantAnalysis(n_components=10)),
        #     ])),
        #     ('interactions', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        #     ('select_features', SelectKBest(score_func=f_regression, k=100)),
        #     ('model', model)
        # ])
    } 