package br.unicamp.ic.execution.knn;

import weka.classifiers.lazy.IBk;

public class BCancerFeatureSelectionExecution extends FeatureSelectionExecution {
	
	public BCancerFeatureSelectionExecution(boolean[] features) {
		super( "breast-cancer.arff", features, 50, 5, 0.01, new IBk());
	}
	
	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true,
				true};
		FeatureSelectionExecution fs = new BCancerFeatureSelectionExecution(features);
		fs.executeAll();
	}
	
	@Override
	public void executeAll() {
		executeFullFeaturesZScore();		
		executeWithZScore();
	}
}
