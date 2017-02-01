package br.unicamp.ic.execution.stacking;

public class ImageSegmentationFeatureSelectionExecution extends FeatureSelectionExecution {

	public ImageSegmentationFeatureSelectionExecution(boolean[] features) {
		super("segment.arff", features, 150, 6, 0.2);
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
