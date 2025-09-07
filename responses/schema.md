Node properties:
- **Show**
  - `embedding_text`: STRING Example: "Norm of the North: King Sized Adventure Before pla"
  - `type`: STRING Available options: ['Movie', 'TV Show']
  - `rating`: STRING Example: "TV-PG"
  - `show_id`: STRING Example: "81145628"
  - `title`: STRING Example: "Norm of the North: King Sized Adventure"
  - `duration`: STRING Example: "90 min"
  - `description`: STRING Example: "Before planning an awesome wedding for his grandfa"
  - `release_year`: INTEGER Min: 1925, Max: 2020
- **Person**
  - `name`: STRING Example: "Richard Finn"
- **Genre**
  - `name`: STRING Example: "Children & Family Movies"
- **Country**
  - `name`: STRING Example: "United States"

Relationship properties:
The relationships:
```
(:Show)-[:PRODUCED_IN]->(:Country)
(:Show)-[:IN_GENRE]->(:Genre)
(:Person)-[:DIRECTED]->(:Show)
(:Person)-[:ACTED_IN]->(:Show)
```
---
---
---
Node properties:
- Question
  - id: INTEGER Min: 71952725, Max: 79724327
  - body: STRING Example: "I am trying to delete all the data points that are"
  - favorite_count: INTEGER Min: 0, Max: 0
  - creation_date: DATE_TIME Min: 2022-04-21T10:13:25Z, Max: 2025-08-03T21:31:42Z
  - score: INTEGER Min: -7, Max: 948
  - title: STRING Example: "Deleting data points in qDrant DB"
  - link: STRING Example: "https://stackoverflow.com/questions/79713858/delet"
- Answer
  - creation_date: DATE_TIME Min: 2022-04-29T11:35:21Z, Max: 2025-08-04T08:16:31Z
  - is_accepted: BOOLEAN 
  - body: STRING Example: "Can you please confirm that req.query.email is n"
  - score: INTEGER Min: -11, Max: 1153
  - id: INTEGER Min: 72057335, Max: 79724626
- User
  - id: INTEGER Min: deleted, Max: 31197606
  - display_name: STRING Example: "Anush"
  - reputation: INTEGER Min: 1, Max: 947179
- Tag
  - name: STRING Example: "javascript"


Relationship properties:
The relationships:
```
(:Question)-[:TAGGED]->(:Tag)
(:Answer)-[:ANSWERS]->(:Question)
(:User)-[:PROVIDED]->(:Answer)
(:User)-[:ASKED]->(:Question)
```