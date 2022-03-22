use sentimentAnalysis

create table trackPackage(
	id int identity(1,1),
	name nvarchar(20),
	time_of_visit datetime
)

create table trackPredictions(
	id int identity(1,1),
	original nvarchar(max),
	sentiment nvarchar(15),
	confidence float,
	intensity nvarchar(30),
	i_proba float,
	subjective nvarchar(30),
	s_proba float,
	time_of_prediction datetime,
)

INSERT INTO trackPredictions(original,sentiment,confidence,intensity,i_proba,subjective,s_proba,time_of_prediction) VALUES(?,?)