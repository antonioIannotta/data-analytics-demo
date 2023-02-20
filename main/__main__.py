import numpy as np
import pandas as pd

print("Data Analytics Demo")

print("Selezionare il tipo di task del quale effettuare la demo: ")
task_selection = input("1: Task 1\n2: Task 2\n3: Task 3")

if int(task_selection) == 1:
    genres = pd.Series(
        ["Adventure", "Animation", "Children", "Comedy", "Fantasy", "Romance", "Drama", "Action", "Crime",
         "Horror", "Mistery", "Sci-Fi", "IMAX", "Documentary", "War", "Musical", "Western", "Film-Noir",
         "(no genres listed)"])

    tag_index_array = []
    for i in range(1, 1129):
        tag_index_array.append(str(i))

    tag_index_array.append("rating")
    tags = pd.Series(np.array(tag_index_array))

    columns = genres.append(tags)

    task_df = pd.DataFrame(columns=columns)
    print(task_df)
