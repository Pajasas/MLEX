all: download perc knn dt
	

download: artificial_separable_test.csv adult.data
	echo download

artificial_separable_test.csv:
	curl http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/classification_data/artificial_objects.tgz >artificial_objects.tgz
	tar xaf artificial_objects.tgz
	rm artificial_objects.tgz

adult.data:
	curl http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data >adult.data
	curl http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test >adult.test

perc:
	echo perc

knn:
	echo knn

dt:
	echo dt
