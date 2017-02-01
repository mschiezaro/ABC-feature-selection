package br.unicamp.ic.execution;

public class LaborFeatureSelectionExecution extends FeatureSelectionExecution {

	public LaborFeatureSelectionExecution(boolean[] features) {
		super("labor.arff", features, 100, 6, 0.1);
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
