package br.unicamp.ic.execution.knn;

import weka.classifiers.lazy.IBk;

public class LaborFeatureSelectionExecution extends FeatureSelectionExecution {

	public LaborFeatureSelectionExecution(boolean[] features) {
		super("labor.arff", features, 100, 10, 0.01, new IBk());
	}
	
	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true};		
		FeatureSelectionExecution fs = new LaborFeatureSelectionExecution(features);
		fs.executeAll();
	}
	
	@Override
	public void executeAll() {
		executeFullFeaturesZScore();		
		executeWithZScore();
	}
}
