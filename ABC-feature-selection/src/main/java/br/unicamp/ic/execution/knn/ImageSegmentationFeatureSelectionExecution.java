package br.unicamp.ic.execution.knn;

import weka.classifiers.lazy.IBk;

public class ImageSegmentationFeatureSelectionExecution extends FeatureSelectionExecution {

	public ImageSegmentationFeatureSelectionExecution(boolean[] features) {
		super("segment.arff", features, 50, 15, 0.05, new IBk());
	}
	
	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true };
		FeatureSelectionExecution fs = new ImageSegmentationFeatureSelectionExecution(features);
		fs.executeAll();
	}
	
	@Override
	public void executeAll() {
		executeFullFeaturesWithNoFilters();		
		executeWithNoFilter();
	}
}
