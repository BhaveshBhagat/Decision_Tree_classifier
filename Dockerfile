From ubuntu

RUN apt-get -y update
RUN apt-get install git
RUN git clone https://github.com/BhaveshBhagat/Decision_Tree_classifier.git
RUN apt-get install python3
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install seaborn

ENTRYPOINT Decision Tree Algo.py
