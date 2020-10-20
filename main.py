# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from stock import stock

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

WALMART = stock('WLMT.csv', 100, "Walmart Stock Price", "12 Oct 2010")
AM_AIRLINES = stock('AAL.csv', 100, "American Airlines Stock Price", "09 Dec 2013")
AMAZON = stock('AMZN.csv', 1500, "Amazon Stock Price", "12 Oct 2010")
KELLOGG = stock('KLLG.csv', 100, "Kellogg Stock Price", "12 Oct 2010")
TRANSOCEAN = stock('RIG.csv', 100, "Transocean Stock Price", "13 Oct 2010")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #WALMART.train_and_graph()
    #AM_AIRLINES.train_and_graph()
    #TRANSOCEAN.train_and_graph()
    AMAZON.train_and_graph()
    #KELLOGG.train_and_graph()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
