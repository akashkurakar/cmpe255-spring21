import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep='\t')
    
    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        return self.chipo.order_id.count()
    
    def info(self) -> None:
        # TODO
        # print data info.
        print(self.chipo.info())
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        return len(self.chipo.columns)
    
    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        for column in self.chipo.columns: 
            print(column) 
    
    def most_ordered_item(self):
        # TODO
        item_name = None
        order_id = -1
        quantity = -1
        itemlist = self.chipo['item_name'].unique()
        items =self.chipo.groupby(['item_name'])['order_id'].sum().sort_values(ascending=False).head(1)
        item_name=items.keys()[0]
        order_id = self.chipo.query(f'item_name=="{item_name}"')['order_id'].sum()
        quantity= items =self.chipo.query(f'item_name=="{item_name}"').groupby(['choice_description'])['quantity'].sum().sort_values(ascending=False).sum()
        return item_name,order_id,quantity

    def total_item_orders(self) -> int:
       # TODO How many items were orderd in total?
       count = 0
       for quantity in self.chipo.quantity:
           count = count + quantity
       return count
   
    def total_sales(self) -> float:
        # TODO 
        # 1. Create a lambda function to change all item prices to float.
        # 2. Calculate total sales.
        total_sales =0.0
        item_price_float = lambda itempricestr: float(itempricestr[1:])
        for i,row in self.chipo.iterrows():
            total_sales =(total_sales+(item_price_float(row.item_price)*row.quantity)) 
        return round(total_sales,2)
   
    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        return self.chipo['order_id'].nunique()
    
    def average_sales_amount_per_order(self) -> float:
        # TODO
        avg = self.total_sales()/self.num_orders()
        return round(avg,2)
    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        return self.chipo['item_name'].nunique()
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        item_df =pd.DataFrame(list(letter_counter.items()),columns=['item_name','quantity']).sort_values('quantity',ascending=False)[:x]
        plot = item_df.plot.bar(x='item_name',y='quantity',rot=0)
        plot.set_xlabel("Items")
        plot.set_ylabel("Number of Orders")
        plot.set_title("Most popular items")
        plt.show(block=True) 
        # TODO
        # 1. convert the dictionary to a DataFrame
        # 2. sort the values from the top to the least value and slice the first 5 items
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        # 5. show the plot. Hint: plt.show(block=True).
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
       # print(pd.Series(np.array(prices)))
        # 1. create a list of prices by removing dollar sign and trailing space.
        # 2. groupby the orders and sum it.
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        item_price = lambda itempricestr:itempricestr.strip()[1:]
        price = self.chipo.groupby(['order_id'])['item_price'].sum()
        prices =[]
        for index,value in price.items():
            l =  {value}
            for value in l:
                m =value.split(" ")
                z=0
                for val in m:
                    n = item_price(val)
                    if n != "" : 
                        z=z+float(n)
                prices.append(round(z,2))
        quantity = self.chipo.groupby(['order_id'])['quantity'].sum()
        plt.xlabel("Order Prices")  
        plt.ylabel("Num Items")  
        plt.title("Numer of items per order price") 
        plt.scatter(x=prices, y=quantity,s=50, c="blue")
        plt.show()
    
        

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926	
    #assert quantity == 159
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()

    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    