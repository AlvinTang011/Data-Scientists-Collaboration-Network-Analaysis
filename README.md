Create conda environment for dependencies:

```
conda env create -f environment.yml
```

Run streamlit to interactively visualize network at (cumulative) year granularity:

```
streamlit run visualize_by_years.py
```

To run the entire analysist:
```
python Data Scientists Collaboration Network Analysist.py
```
Note, if the html file appears wrong link, it is saved within the folder 'Results'. Open to view the network
All images for properties are output as pdf except the network which is in html for interactivity
User input for max degree node in integer to obtain desired K(max) for network

For Input file, ensure that it follows the format established in [DataScientists.xls file](https://github.com/AlvinTang011/Data-Scientists-Collaboration-Network-Analaysis/blob/main/Input/DataScientists.xls) - name, country, institution, dblp, expertise


The output file will include every analysis used to identify the relationships between the data scientists within the given input file

An analysis of the current relationship of data scientists is done up in the report
- [Network Science-Based Analysis of Collaboration Networks of Data Scientists.pdf](https://github.com/AlvinTang011/Data-Scientists-Collaboration-Network-Analaysis/blob/main/Network%20Science-Based%20Analysis%20of%20Collaboration%20Networks%20of%20Data%20Scientists.pdf)
