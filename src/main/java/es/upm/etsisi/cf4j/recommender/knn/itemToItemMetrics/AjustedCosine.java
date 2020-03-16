package es.upm.etsisi.cf4j.recommender.knn.itemToItemMetrics;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.Item;
import es.upm.etsisi.cf4j.data.User;

/**
 * Implements traditional Ajusted Cosine as CF similarity metric for the items.
 * 
 * @author Fernando Ortega
 */
public class AjustedCosine extends ItemToItemMetric {

	public AjustedCosine(DataModel datamodel, double[][] similarities) {
		super(datamodel, similarities);
	}

	@Override
	public double similarity(Item item, Item otherItem) {
		
		int u = 0, v = 0, common = 0; 
		double num = 0d, denActive = 0d, denTarget = 0d;

		while (u < item.getNumberOfRatings() && v < otherItem.getNumberOfRatings()) {
			if (item.getUser(u) < otherItem.getUser(v)) {
				u++;
			} else if (item.getUser(u) > otherItem.getUser(v)) {
				v++;
			} else {
				int userIndex = item.getUser(u);
				User user = super.datamodel.getUserAt(userIndex);
				double avg = user.getRatigAverage();
				
				double fa = item.getRating(u) - avg;
				double ft = otherItem.getRating(v) - avg;
				
				num += fa * ft;
				denActive += fa * fa;
				denTarget += ft * ft;
				
				common++;
				u++; 
				v++;
			}	
		}

		// If there is not ratings in common, similarity does not exists
		if (common == 0) return Double.NEGATIVE_INFINITY;

		// Denominator can not be zero
		if (denActive == 0 || denTarget == 0) return Double.NEGATIVE_INFINITY;

		// Return similarity
		return num / Math.sqrt(denActive * denTarget);
	}
}