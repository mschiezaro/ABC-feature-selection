package br.unicamp.ic.execution;

public class BCancerFeatureSelectionExecution extends FeatureSelectionExecution {
	
	public BCancerFeatureSelectionExecution(boolean[] features) {
		super( "breast-cancer.arff", features, 100, 6, 0.1);
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
