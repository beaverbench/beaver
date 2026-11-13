omni_sql_input_prompt_template = '''Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
{db_engine}

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.
'''

class Prompts:
    def __init__(self):
        pass
    def get_condition_onmit_tables(self):
        return ["-- Include all", "-- Omit", "-- Continue", "-- Union all", "-- ...", "-- List all", "-- Replace this", "-- Each table", "-- Add other"]
    def get_prompt_dialect_list_all_tables(self, table_struct, api):
        if api == "snowflake":
            return f"When performing a UNION operation on many tables, ensure that all table names are explicitly listed. Union first and then add condition and selection. e.g. SELECT \"col1\", \"col2\" FROM (TABLE1 UNION ALL TABLE2) WHERE ...; Don't write sqls as (SELECT col1, col2 FROM TABLE1 WHERE ...) UNION ALL (SELECT col1, col2 FROM TABLE2 WHERE ...); Don't use {self.get_condition_onmit_tables()} to omit any table. Table names here: {table_struct}\n"
        elif api == "bigquery":
            return "When performing a UNION operation on many tables with similar prefix, you can use a wildcard table to simplify your query. e.g., SELECT col1, col2 FROM `project_id.dataset_id.table_prefix*` WHERE _TABLE_SUFFIX IN ('table1_suffix', 'table2_suffix');. Avoid manually listing tables unless absolutely necessary.\n"
        else:
            return ""
    def get_prompt_fuzzy_query(self):
        return "For string-matching scenarios, if the string is decided, don't use fuzzy query. e.g. Get the object's title contains the word \"book\"\nHowever, if the string is not decided, you may use fuzzy query and ignore upper or lower case. e.g. Get articles that mention \"education\".\n"
    def get_prompt_decimal_places(self):
        return "If the task description does not specify the number of decimal places, retain all decimals to four places.\n"
    def get_prompt_convert_symbols(self):
        return "For string-matching scenarios, convert non-standard symbols to '%'. e.g. ('he’s to he%s)\n"
    def get_prompt_knowledge(self):
        return "Your knowledge is based on information in database. Don't use your own knowledge.\n"
    def get_prompt_dialect_nested(self, api):
        if api == "snowflake":
            return "For columns in json nested format: e.g. SELECT t.\"column_name\", f.value::VARIANT:\"key_name\"::STRING AS \"abstract_text\" FROM PATENTS.PATENTS.PUBLICATIONS t, LATERAL FLATTEN(input => t.\"json_column_name\") f; DO NOT directly answer the task and ensure all column names are enclosed in double quotations. For nested columns like event_params, when you don't know the structure of it, first watch the whole column: SELECT f.value FROM table, LATERAL FLATTEN(input => t.\"event_params\") f;\n"
        elif api == "bigquery":
            return "Extract a specific key from a nested JSON column: SELECT t.\"column_name\", JSON_EXTRACT_SCALAR(f.value, \"$.key_name\") AS \"abstract_text\" FROM `database.schema.table` AS t, UNNEST(JSON_EXTRACT_ARRAY(t.\"json_column_name\")) AS f;\nWhen the structure of the nested column (e.g., event_params) is unknown, first inspect the whole column: SELECT f.value FROM `project.dataset.table` AS t, UNNEST(JSON_EXTRACT_ARRAY(t.\"event_params\")) AS f;\n"
        elif api == "sqlite":
            return "Extract a specific key from a nested JSON column: SELECT t.\"column_name\", json_extract(f.value, '$.key_name') AS \"abstract_text\" FROM \"table_name\" AS t, json_each(t.\"json_column_name\") AS f;\nWhen the structure of the nested column (e.g., event_params) is unknown, first inspect the whole column: SELECT f.value FROM \"table_name\" AS t, json_each(t.\"event_params\") AS f;\n"
        elif api == "mysql":
            return "For JSON columns in MySQL: Use JSON_EXTRACT to extract values: SELECT JSON_EXTRACT(`json_column`, '$.key_name') AS `extracted_value` FROM `table_name`; For nested arrays, use JSON_TABLE to unnest: SELECT t.`column_name`, jt.`key_name` FROM `table_name` AS t, JSON_TABLE(t.`json_column`, '$[*]' COLUMNS(`key_name` VARCHAR(255) PATH '$.key_name')) AS jt;\n"
        else:
            return "Unsupported API. Please provide a valid API name ('snowflake', 'bigquery', 'sqlite', 'mysql')."
    def get_prompt_dialect_basic(self, api):
        if api == "snowflake":
            return "```sql\nSELECT \"COLUMN_NAME\" FROM DATABASE.SCHEMA.TABLE WHERE ... ``` (Adjust \"DATABASE\", \"SCHEMA\", and \"TABLE\" to match actual names, ensure all column names are enclosed in double quotations)"
        elif api == "bigquery":
            return "```sql\nSELECT `column_name` FROM `database.schema.table` WHERE ... ``` (Replace `database`, `schema`, and `table` with actual names. Enclose column names and table identifiers with backticks.)"
        elif api == "sqlite":
            return "```sql\nSELECT DISTINCT \"column_name\" FROM \"table_name\" WHERE ... ``` (Replace \"table_name\" with the actual table name. Enclose table and column names with double quotations if they contain special characters or match reserved keywords.)"
        elif api == "mysql":
            return "```sql\nSELECT DISTINCT `column_name` FROM `table_name` WHERE ... ``` (Replace `table_name` with the actual table name. Enclose table and column names with backticks.)"
        else:
            raise NotImplementedError("Unsupported API. Please provide a valid API name ('snowflake', 'bigquery', 'sqlite', 'mysql').")
    def get_prompt_dialect_string_matching(self, api):
        if api == "snowflake":
            return "Don't directly match strings if you are not convinced. Use fuzzy query first: WHERE str ILIKE \"%target_str%\" For string matching, e.g. meat lovers, you should use % to replace space. e.g. ILKIE %meat%lovers%.\n"
        elif api == "bigquery":
            return "Don't directly match strings if you are not convinced. Use LOWER for fuzzy queries: WHERE LOWER(str) LIKE LOWER('%target_str%'). For example, to match 'meat lovers', use LOWER(str) LIKE '%meat%lovers%'.\n"
        elif api == "sqlite":
            return "Don't directly match strings if you are not convinced. For fuzzy queries, use: WHERE str LIKE '%target_str%'. For example, to match 'meat lovers', use WHERE str LIKE '%meat%lovers%'. If case sensitivity is needed, add COLLATE BINARY: WHERE str LIKE '%target_str%' COLLATE BINARY.\n"
        elif api == "mysql":
            return "Don't directly match strings if you are not convinced. For fuzzy queries, use: WHERE str LIKE '%target_str%'. For example, to match 'meat lovers', use WHERE str LIKE '%meat%lovers%'. For case-insensitive matching, use: WHERE LOWER(str) LIKE LOWER('%target_str%').\n"
        else:
            raise NotImplementedError("Unsupported API. Please provide a valid API name ('snowflake', 'bigquery', 'sqlite', 'mysql').")

    def get_format_prompt(self):
        format_prompt = "This is an SQL task. Please provide the simplest possible answer format in ```csv``` format like a table.\n"
        format_prompt += "e.g.1. Including the travel coordinates and the cumulative travel distance at each point. Format: ```csv\ntravel_coordinates,cumulative_travel_distance\nPOINT(longitude1 latitude1),distance1:int\nPOINT(longitude2 latitude2),distance2:int\n...```\n"
        format_prompt += "When asked something without specifying name or id, provide both. e.g.2. Which products had a seasonality-adjusted sales ratio that stayed consistently above 2 for every month in the year 2017? Format: ```csv\nproduct_name,product_id\nproduct_name1:str,product_id1:int\n...```\n"
        format_prompt += "Do not output any SQL queries.\n"
        return format_prompt

    def get_exploration_prompt(self, api, table_struct):
        exploration_prompt = f"Write at most 10 {api} simple SQL queries in format like:\n {self.get_prompt_dialect_basic(api)}\nin ```sql``` code block to have an understanding of values in related columns.\n"
        exploration_prompt += "Each query should be different. Don't query about any SCHEMA or checking data types. You can write SELECT query only. Try to use DISTINCT. Don't output the final answer.\n"
        exploration_prompt += "Write annotations to describe each SQL, format like ```sql\n-- Description: \n```.\n"

        # exploration_prompt += "When exploring a table, first generate a SQL query to view 5 distinct rows, then generate another SQL query with appropriate conditions.\n"

        exploration_prompt += self.get_prompt_dialect_nested(api)
                
        # exploration_prompt += self.get_prompt_convert_symbols()
        
        # exploration_prompt += self.get_prompt_dialect_string_matching(api)
        
        # exploration_prompt += "For time-related queries, given the variety of formats, avoid using time converting functions unless you are certain of the specific format being used.\n"
        
        # exploration_prompt += "When generating SQLs, be aware of quotation matching: 'Vegetarian\"; You sometimes match \' with \" which may cause an error.\n"

        # exploration_prompt += f"You can only use tables in {table_struct}"
        
        # exploration_prompt += self.get_prompt_knowledge()

        return exploration_prompt

    def get_exploration_refine_prompt(self, sql, corrected_sql, sqls, res):
        return f"```sql\n{sql}``` is corrected to ```sql\n{corrected_sql}```. And the result is: \n{res}\n Please correct other sqls based on results if they have similar errors. Otherwise don't modify the SQL. SQLs: {sqls}. For each SQL, answer in ```sql\n-- Description: \n``` format.\n"

    def get_exploration_self_correct_prompt(self, sql, error):
        return f"Input sql:\n{sql}\nThe error information is:\n" + str(error) + "\nPlease correct it based on previous context and output the thinking process with only one sql query in ```sql\n-- Description: \n``` format. Don't just analyze without SQL or output several SQLs.\n"

    # def get_beaver_one_shot_example(self):
    #     """
    #     One-shot example for Beaver + MySQL, used as an in-context demonstration
    #     when generating SQL for Beaver (especially option 4).
    #     """
    #     example_text = ""
    #     example_text += "[One-shot Example]\n"
    #     example_text += "Example question:\n"
    #     example_text += (
    #         "List the unique full room names, their corresponding building names, street "
    #         "addresses, cities, states, postal codes, and building heights for rooms associated "
    #         "with subjects that Computer Science students can enroll in.\n"
    #     )
    #     example_text += "\nExample SQL (MySQL dialect):\n"
    #     example_text += "```sql\n"
    #     example_text += (
    #         "SELECT DISTINCT\n"
    #         "  fr.ROOM_FULL_NAME,\n"
    #         "  b.BUILDING_NAME,\n"
    #         "  b.BUILDING_STREET_ADDRESS,\n"
    #         "  fba.CITY,\n"
    #         "  fba.STATE,\n"
    #         "  fba.POSTAL_CODE,\n"
    #         "  fb.BUILDING_HEIGHT\n"
    #         "FROM COURSE_CATALOG_SUBJECT_OFFERED AS cso\n"
    #         "JOIN FCLT_ROOMS AS fr\n"
    #         "  ON cso.MEET_PLACE = fr.FCLT_ROOM_KEY\n"
    #         "JOIN FCLT_BUILDING_ADDRESS AS fba\n"
    #         "  ON fba.FCLT_BUILDING_KEY = fr.FCLT_BUILDING_KEY\n"
    #         "JOIN FCLT_BUILDING AS fb\n"
    #         "  ON fb.FCLT_BUILDING_KEY = fr.FCLT_BUILDING_KEY\n"
    #         "JOIN BUILDINGS AS b\n"
    #         "  ON b.BUILDING_KEY = fr.FCLT_BUILDING_KEY\n"
    #         "WHERE cso.DEPARTMENT_NAME = 'Electrical Eng & Computer Sci'\n"
    #         "  AND fba.ADDRESS_PURPOSE = 'STREET';\n"
    #     )
    #     example_text += "```\n"
    #     example_text += (
    #         "This example shows how to:\n"
    #         "- Join course offerings to rooms, building address, and building metadata.\n"
    #         "- Filter by a specific department name and address purpose.\n"
    #         "- Select only the columns requested in the question.\n"
    #     )
    #     return example_text


    def get_beaver_one_shot_example(self, db_id="dw", option=2):
        """
        One-shot example for Beaver + MySQL, used as an in-context demonstration
        when generating SQL for Beaver.
        """
        if db_id == "sp":
            return self.get_sp_one_shot_example(option)
        elif db_id == "neutron":
            return self.get_neutron_one_shot_example(option)
        elif db_id == "nova":
            return self.get_nova_one_shot_example(option)
        return self.get_dw_one_shot_example(option)

    def get_sp_one_shot_example(self, option=2):
        txt = r'''[One-shot Example]
Database ID: sp

[Database description]
CREATE TABLE room_properties (
    ROOM_ID VARCHAR(20) -- example: ['000', '000', '107'],
    ROOMTYPE VARCHAR(20) -- example: ['COMMOM', 'COMMOM', 'APARTMENT'],
    ROOMPHONE VARCHAR(15) -- example: ['(617) 225-9610', '(617) 225-8666', '(617) 258-0046'],
    OFFICER_ID VARCHAR(30) -- example: ['sp-room-chair', 'sp-room-chair', 'sp-room-chair'],
    SQFT INT -- example: ['626', '626', '602'],
    CAPACITY INT -- example: ['70', '70', '4'],
    ACCESSIBLE VARCHAR(4) -- example: ['YES', 'YES', 'YES'],
    ROOMSUBTYPE VARCHAR(40) -- example: ['Kitchen', 'Multipurpose Room', '4-person/2-bedroom'],
    ACTIVE VARCHAR(4) -- example: ['YES', 'YES', 'YES'],
    CORE VARCHAR(20) -- example: ['0', '0', 'KAA'],
    KEYCODE VARCHAR(8) -- example: ['KAT', 'KBC', 'KBH'],
    NOTE TEXT -- example: ['formerly House Common room', 'formerly House Common room', 'formerly House Common room'],
    ASSIGNABLE TINYINT(1) -- example: ['0', '0', '0']
);

CREATE TABLE publicity (
    ID INT -- example: ['1244', '1245', '1246'],
    GOOGLECAL_EVENTID VARCHAR(255) -- example: ['galgtu508al4aroa0r16r80jlc', '61c6jokb6q88jphd1elgodfj0g', '4a985ro90209nsq2e7gbak46eg'],
    ATHENAUSERNAME VARCHAR(128) -- example: ['wenliu', 'wenliu', 'wenliu'],
    EVENT_ID VARCHAR(255) -- example: ['!(()&&!|*|*|', '"+response.write(9033242*9911871)+"', '"+response.write(9078404*9686550)+"'],
    EVENT_START DATETIME -- example: ['None', 'None', 'None'],
    EVENT_START_TIME TIME -- example: ['61200.0', '0.0', '39600.0'],
    EVENT_START_DATE DATE -- example: ['2008-07-21', '2008-07-21', '2008-07-21'],
    EVENT_END DATETIME -- example: ['2008-07-21T20:00:00', '2008-07-21T00:00:00', '2008-07-21T00:00:00'],
    EVENT_END_TIME TIME -- example: ['72000.0', '0.0', '57600.0'],
    EVENT_END_DATE DATE -- example: ['2008-07-21', '2008-07-21', '2008-07-21'],
    EVENT_TITLE VARCHAR(255) -- example: ['Hall Councilor BBQ', 'Magic Night', 'Eating Clubs @ Sidney Pacific'],
    EVENT_DESCRIPTION TEXT -- example: ['Come out and enjoy food with some friends!', '', 'Are you tired of the food at the Student Center and Stata Center? Are you nostalgic for some home cooking? Do you have a new recipe that you are dying to try? Would you like to meet new friends with similar culinary tastes? Do you enjoy a good quality, yet inexpensive meal? If so, you should join (or perhaps create) an eating club!'],
    EVENT_LOCATION VARCHAR(255) -- example: ['', '', ''],
    EVENT_SPONSORS VARCHAR(255) -- example: ['', '', ''],
    EVENT_EXTERNAL_PUBLICITY TINYINT(1) -- example: ['0', '0', '1'],
    EVENT_AFFILIATION VARCHAR(255) -- example: ['', '', ''],
    EVENT_TYPE VARCHAR(255) -- example: ['event', 'event', 'anno'],
    EVENT_RECURRENCE VARCHAR(128) -- example: ['', '', ''],
    EMAIL VARCHAR(128) -- example: ['sp-vp-info', 'sp-vp-info', 'sp-vp-info'],
    POSTER_POSTBYDATE DATE -- example: ['2008-07-21', '2008-07-21', '2008-07-28'],
    POSTER_DESIGN TEXT -- example: ['', '', ''],
    POSTER_REQUESTDESIGN TINYINT(1) -- example: ['0', '1', '0'],
    POSTER_URL VARCHAR(255),
    POSTER_FILENAME VARCHAR(255) -- example: ['Hall-Councilor-BBQ-Flyer[2]-1216680852.png', 'August Brunch-1217272904.pdf', 'AV Co-Chair Flyer-1217273240.ai'],
    POSTER_WEB_FILENAME VARCHAR(255) -- example: ['Hall-Councilor-BBQ-Flyer[3]-1216681197.png', 'Magic-Night-Flyer[1]-1216669047.png', 'beach_trip-1216951619.jpg'],
    POSTER_COMPLETED TINYINT(1) -- example: ['1', '1', '0'],
    POSTER_MODIFIED TINYINT(1) -- example: ['1', '1', '0'],
    ROOM_START DATETIME -- example: ['2008-07-21T15:00:00', '2008-08-08T19:30:00', '2008-08-25T20:00:00'],
    ROOM_START_TIME TIME -- example: ['54000.0', '70200.0', '72000.0'],
    ROOM_START_DATE DATE -- example: ['2008-07-21', '2008-08-08', '2008-08-25'],
    ROOM_END DATETIME -- example: ['2008-07-21T21:00:00', '2008-08-08T22:00:00', '2008-08-25T22:00:00'],
    ROOM_END_TIME TIME -- example: ['75600.0', '79200.0', '79200.0'],
    ROOM_END_DATE DATE -- example: ['2008-07-21', '2008-08-08', '2008-08-25'],
    ROOM_AVSUPPORT TINYINT(1) -- example: ['0', '1', '1'],
    ROOM_AVDESCRIPTION TEXT -- example: ['', 'Satellite TV hookup and sound', 'Live band will be there.  Microphones and microphone-stands might be needed.  Two speakers can be used.'],
    ROOM_ID VARCHAR(255) -- example: ['157,159,160', '185', '163'],
    ROOM_APPROVAL TINYINT(1) -- example: ['1', '1', '1'],
    ROOM_USAGE VARCHAR(255) -- example: ['Government Related', 'Social Event', 'Social Event'],
    ROOM_RECURRENCE VARCHAR(128) -- example: ['', '', ''],
    AV_APPROVAL TINYINT(1),
    ROOM_MODIFIED TINYINT(1) -- example: ['0', '0', '0'],
    ANNO_DESCRIPTION TEXT -- example: ['', '', ''],
    ANNO_URL VARCHAR(255) -- example: ['http://www.nytimes.com', '', ''],
    ANNO_URLTEXT VARCHAR(255) -- example: ['Click here!', '', ''],
    ANNO_APPROVAL TINYINT(1) -- example: ['1', '1', '1'],
    ANNO_MODIFIED TINYINT(1) -- example: ['0', '0', '0'],
    ANNO_TYPE VARCHAR(255) -- example: ['Internal', 'Internal', 'Internal'],
    SPTV_URL VARCHAR(255),
    SPTV_FILENAME VARCHAR(255) -- example: ['Hall-Councilor-BBQ-Flyer[1]-1216680887.png', 'White-Mountains-Camping-Trip-Slide-1217569241.png', 'fells-1217548734.png'],
    SPTV_COMMENTS TEXT -- example: ['', 'test', ''],
    SPTV_DISPLAY_SECONDS VARCHAR(10) -- example: ['10', '10', '10'],
    SPTV_END_DATE DATE -- example: ['2008-07-22', '2008-08-10', '2008-08-16'],
    SPTV_START_DATE DATE -- example: ['2008-07-21', '2008-07-29', '2008-07-31'],
    SPTV_COMPLETED TINYINT(1) -- example: ['1', '1', '1'],
    SPTV_MODIFIED TINYINT(1) -- example: ['1', '0', '0'],
    RSVP_ID VARCHAR(128) -- example: ['479', '465', ''],
    RSVP_DESCRIPTION TEXT -- example: [''],
    SUBMISSIONDATE DATETIME -- example: ['2008-07-21T19:14:00', '2008-07-21T19:01:00', '2008-07-21T15:57:00'],
    MODIFIED TINYINT(1) -- example: ['1', '1', '1'],
    METHODS VARCHAR(128) -- example: ['anno#web#poster#room#sptv#', 'web#poster#', 'web#'],
    OFFICER_COMMENTS TEXT -- example: ['', '', ''],
    EXTERNALEVENT TINYINT(1) -- example: ['0', '0', '0'],
    WEB_DESCRIPTION TEXT -- example: ['Come out and enjoy food with some friends!', '<A HREF=http://www.flickr.com/photos/sidpac/show/with/2622900525/>Event Photo Gallery</a> <p>SP Magic Night with world renowned magician took place in the Multipurpose room on Sunday June 29th. We had a packed house for this event, where everyone enjoyed some light snacks and an hour of up-close magic with David Hall.</p> <p>Special thanks to our social chairs William Loh and Kaveh Milaninia for organizing the event and thanks to all the helpers who came down early to help setup.</p> <p>If you enjoyed this event and would like to give us your comments, please e-mail sp-social-chair[at]mit.edu.</p> ', 'Are you tired of the food at the Student Center and Stata Center? Are you nostalgic for some home cooking? Do you have a new recipe that you are dying to try? Would you like to meet new friends with similar culinary tastes? Do you enjoy a good quality, yet inexpensive meal? If so, you should join (or perhaps create) an eating club! Click <A HREF=http://s-p.mit.edu/EatingClub>here</a> for details!'],
    WEB_APPROVAL TINYINT(1) -- example: ['0', '0', '0'],
    WEB_MODIFIED TINYINT(1) -- example: ['0', '0', '0'],
    SPTV_IMPORTANT TINYINT(1) -- example: ['0', '0', '0'],
    SPTV_INCREASING TINYINT(1) -- example: ['0', '0', '0'],
    TIME_INSENSITIVE TINYINT(1) -- example: ['0', '0', '0'],
    EXTERNALPOSTED TINYINT(1) -- example: ['0', '0', '0'],
    REQUEST_PHOTOCHAIR TINYINT(1) -- example: ['0', '0', '0'],
    PHOTOCHAIR_APPROVAL TINYINT(1) -- example: ['0', '0', '0'],
    ROOM_EXTERNALATTENDEESEXPECTED TINYINT(1) -- example: ['1', '1', '1'],
    INITIALSUBMISSIONTIMESTAMP TIMESTAMP -- example: ['None', 'None', 'None']
);

CREATE TABLE room_history (
    ROOMHISTORYENTRY_ID INT -- example: ['12811', '665', '1828'],
    MIT_ID INT -- example: ['0', '12345', '222222115'],
    ROOM_ID VARCHAR(20) -- example: ['000', '000', '107'],
    MOVEINALLOWEDDATE INT -- example: ['20020824', '20020824', '20020816'],
    CHECKINDATE INT -- example: ['20020824', '20020828', '20020816'],
    MOVEOUTREQUIREDDATE INT -- example: ['20030815', '20040831', '20040831'],
    CHECKOUTDATE INT -- example: ['20020920', '20040831', '20040803'],
    ASSIGNED_OFFICER_ID VARCHAR(30) -- example: ['sp-room-chair', 'sp-room-chair', 'sp-room-chair'],
    CHECKINACCESS_ID VARCHAR(50) -- example: ['David Varisco', 'David Varisco', 'Dennis Meaney'],
    CHECKINNOTE TEXT -- example: ['', '', ''],
    CHECKOUTACCESS_ID VARCHAR(50) -- example: ['Dennis Collins', 'Dennis J Collins: sp-housemanager', 'David Varisco: sp-deskworker'],
    CHECKOUTNOTE TEXT -- example: ['', '', ''],
    SUBLESSEE TINYINT(1) -- example: ['0', '0', '0']
);

-- Schema Mapping (question concepts to columns):
-- 'room ID' -> room_properties.ROOM_ID
-- 'room type' -> room_properties.ROOMTYPE
-- 'events' -> publicity.EVENT_ID
-- 'event duration' -> publicity.EVENT_END, publicity.EVENT_START
-- 'residents' -> room_history.MIT_ID

You should use the provided mapping to determine which columns and tables should be used in the SQL statement.

-- Join Keys (how tables connect):
-- room_properties.ROOM_ID = publicity.ROOM_ID
-- room_properties.ROOM_ID = room_history.ROOM_ID

You should use the provided join keys to determine how to connect the tables in the SQL statement.

Example question:
For rooms that have hosted more than 5 events, list the room ID, room type, the total number of events, the average event duration, and the number of residents who have ever lived in that room.

Example SQL (MySQL dialect):
```sql
SELECT rp.Room_ID, rp.RoomType, COUNT(pe.Event_ID) AS total_events, AVG(DATEDIFF(pe.Event_End, pe.Event_Start)) AS avg_event_duration, COUNT(DISTINCT rh.MIT_ID) AS total_residents FROM room_properties rp JOIN publicity pe ON rp.Room_ID = pe.Room_ID LEFT JOIN room_history rh ON rp.Room_ID = rh.Room_ID GROUP BY rp.Room_ID, rp.RoomType HAVING total_events > 5;
```
'''
        if option == 2:
            import re
            txt = re.sub(r'-- Schema Mapping \(question concepts to columns\):.*SQL statement\.\s+', '', txt, flags=re.DOTALL)
            txt = re.sub(r'-- Join Keys \(how tables connect\):.*SQL statement\.\s+', '', txt, flags=re.DOTALL)
        return txt

    def get_neutron_one_shot_example(self, option=2):
        txt = r'''[One-shot Example]
Database ID: neutron

[Database description]
CREATE TABLE qos_port_policy_bindings (
    policy_id VARCHAR(36) -- example: ['0366a68f-a9cb-4c9f-8646-9293144a1da6', '0366a68f-a9cb-4c9f-8646-9293144a1da6', '0366a68f-a9cb-4c9f-8646-9293144a1da6'],
    port_id VARCHAR(36) -- example: ['086be785-3bc4-4712-ba3b-da0f4c071d37', '18678248-be20-410c-99f5-1779813583c7', '1d262b9f-1d45-42f0-9755-92762283a005']
);

CREATE TABLE ports (
    project_id VARCHAR(255) -- example: ['d1fe1eee802541b8bd5ed61cd3a05d77', 'd1fe1eee802541b8bd5ed61cd3a05d77', 'd1fe1eee802541b8bd5ed61cd3a05d77'],
    id VARCHAR(36) -- example: ['000e3eac-b72e-436f-8703-a2b8e390c500', '00201977-8051-4089-a352-25e408ec211e', '003260c6-932f-4100-afab-657d47648356'],
    name VARCHAR(255) -- example: ['', '', ''],
    network_id VARCHAR(36) -- example: ['0a1d0a27-cffa-4de3-92c5-9d3fd3f2e74d', '0a1d0a27-cffa-4de3-92c5-9d3fd3f2e74d', '0a1d0a27-cffa-4de3-92c5-9d3fd3f2e74d'],
    mac_address VARCHAR(32) -- example: ['fa:16:3e:7a:5e:39', 'fa:16:3e:09:a5:50', 'fa:16:3e:f3:b2:01'],
    admin_state_up TINYINT(1) -- example: ['1', '1', '1'],
    status VARCHAR(16) -- example: ['DOWN', 'DOWN', 'DOWN'],
    device_id VARCHAR(255) -- example: ['509284', '509070', '509285'],
    device_owner VARCHAR(255) -- example: ['compute:nova', 'compute:nova', 'compute:nova'],
    standard_attr_id BIGINT -- example: ['116', '118', '122'],
    ip_allocation VARCHAR(16) -- example: ['immediate', 'immediate', 'immediate']
);

CREATE TABLE ml2_dvr_port_bindings (
    port_id VARCHAR(36) -- example: ['000e3eac-b72e-436f-8703-a2b8e390c500', '00201977-8051-4089-a352-25e408ec211e', '003260c6-932f-4100-afab-657d47648356'],
    host VARCHAR(255) -- example: ['sonic-95', 'grav6-70', 'layer-89'],
    router_id VARCHAR(36),
    vif_type VARCHAR(64) -- example: ['ovs', 'ovs', 'ovs'],
    vif_details VARCHAR(4095) -- example: ['{"port_filter": true, "ovs_hybrid_plug": true}', '{"port_filter": true, "ovs_hybrid_plug": true}', '{"port_filter": true, "ovs_hybrid_plug": true}'],
    vnic_type VARCHAR(64) -- example: ['normal', 'normal', 'normal'],
    profile VARCHAR(4095) -- example: ['{}', '{}', '{}'],
    status VARCHAR(16) -- example: ['DOWN', 'DOWN', 'DOWN']
);

-- Schema Mapping (question concepts to columns):
-- 'number of ports' -> qos_port_policy_bindings.port_id
-- 'ports bound to a QoS policy' -> qos_port_policy_bindings.port_id
-- 'DVR port binding' -> ml2_dvr_port_bindings.port_id
-- 'port administratively up' -> ports.admin_state_up
-- 'DVR binding status DOWN' -> ml2_dvr_port_bindings.status

You should use the provided mapping to determine which columns and tables should be used in the SQL statement.

-- Join Keys (how tables connect):
-- qos_port_policy_bindings.port_id = ports.id
-- ports.id = ml2_dvr_port_bindings.port_id

You should use the provided join keys to determine how to connect the tables in the SQL statement.

Example question:
Count the number of ports on network ID `0a1d0a27-cffa-4de3-92c5-9d3fd3f2e74d` that are bound to a QoS policy and also have a DVR port binding where the port is administratively up and the DVR binding status is DOWN.

Example SQL (MySQL dialect):
```sql
SELECT COUNT(*) AS qos_bound_admin_up_down_dvr_ports FROM qos_port_policy_bindings qppb JOIN ports p ON qppb.port_id = p.id JOIN ml2_dvr_port_bindings dvr ON p.id = dvr.port_id WHERE p.admin_state_up = 1 AND dvr.status = 'DOWN' AND p.network_id IN ('0a1d0a27-cffa-4de3-92c5-9d3fd3f2e74d');
```
'''
        if option == 2:
            import re
            txt = re.sub(r'-- Schema Mapping \(question concepts to columns\):.*SQL statement\.\s+', '', txt, flags=re.DOTALL)
            txt = re.sub(r'-- Join Keys \(how tables connect\):.*SQL statement\.\s+', '', txt, flags=re.DOTALL)
        return txt

    def get_nova_one_shot_example(self, option=2):
        txt = r'''[One-shot Example]
Database ID: nova

[Database description]
CREATE TABLE instance_system_metadata (
    created_at DATETIME -- example: ['2014-10-15T01:44:40', '2014-10-15T01:44:40', '2014-10-15T01:44:40'],
    updated_at DATETIME,
    deleted_at DATETIME,
    id INT -- example: ['14', '15', '16'],
    instance_uuid VARCHAR(36) -- example: ['891365b3-6b62-4e92-b683-1694678b9e74', '891365b3-6b62-4e92-b683-1694678b9e74', '891365b3-6b62-4e92-b683-1694678b9e74'],
    key VARCHAR(255) -- example: ['image_instance_type_root_gb', 'image_image_location', 'image_image_type'],
    value VARCHAR(255) -- example: ['arrow_theta', 'neon', 'neon'],
    deleted INT -- example: ['0', '0', '0']
);

CREATE TABLE instances (
    created_at DATETIME -- example: ['2014-10-15T01:44:40', '2014-10-19T01:38:54', '2014-10-19T02:02:46'],
    updated_at DATETIME -- example: ['2023-01-31T16:05:25', '2023-01-31T16:04:32', '2023-01-31T16:03:41'],
    deleted_at DATETIME,
    id INT -- example: ['509070', '509284', '509285'],
    internal_id INT,
    user_id VARCHAR(255) -- example: ['be417436c0174498bf721e849e119555', '79081575ea104d4cbfb9bbb8254484d6', '79081575ea104d4cbfb9bbb8254484d6'],
    project_id VARCHAR(255) -- example: ['d1fe1eee802541b8bd5ed61cd3a05d77', '05001dd7e5be45ef8b1c7a54fa56c15e', '05001dd7e5be45ef8b1c7a54fa56c15e'],
    image_ref VARCHAR(255) -- example: ['e533ca5e-1dc0-4b4d-8023-bc15624a5e63', '9cb774b2-4659-4a93-b68e-b9330fd3eb5f', 'eaa8a5a2-1082-48e4-9f7d-e7be43b55c44'],
    kernel_id VARCHAR(255),
    ramdisk_id VARCHAR(255),
    launch_index INT -- example: ['0', '0', '0'],
    key_name VARCHAR(255) -- example: ['solar_plasm', 'credo-glyph', 'credo-glyph'],
    key_data TEXT -- example: ['2f6cbc63d7af29b70f7d341260041795', 'c2ccf6a097f9b850b8ee26009533cf7e', 'c2ccf6a097f9b850b8ee26009533cf7e'],
    power_state INT -- example: ['4', '4', '4'],
    vm_state VARCHAR(255) -- example: ['shelved_offloaded', 'shelved_offloaded', 'shelved_offloaded'],
    memory_mb INT -- example: ['32768', '16384', '16384'],
    vcpus INT -- example: ['8', '16', '16'],
    hostname VARCHAR(255) -- example: ['glyph-helix-nexus', 'flux-flare', 'orbit-nexus-mover'],
    host VARCHAR(255) -- example: ['', '', ''],
    user_data TEXT -- example: ['replaced_user_data.509070', 'replaced_user_data.509284', 'replaced_user_data.509285'],
    reservation_id VARCHAR(255) -- example: ['r-m3yx01v5', 'r-sejmfyws', 'r-svtnrz0j'],
    scheduled_at DATETIME -- example: ['2014-10-15T01:44:41', '2014-10-19T01:38:55', '2014-10-19T02:02:47'],
    launched_at DATETIME -- example: ['2016-09-26T17:42:44', '2016-09-26T15:31:35', '2016-09-26T15:24:01'],
    terminated_at DATETIME,
    display_name VARCHAR(255) -- example: ['forge', 'strat', 'forge'],
    display_description VARCHAR(255) -- example: ['forge', 'strat', 'forge'],
    availability_zone VARCHAR(255) -- example: ['flare3', 'flare3', 'flare3'],
    locked TINYINT(1) -- example: ['0', '0', '0'],
    os_type VARCHAR(255),
    launched_on TEXT -- example: ['grav6-70', 'sonic-95', 'layer-89'],
    instance_type_id INT -- example: ['81', '69', '69'],
    vm_mode VARCHAR(255),
    uuid VARCHAR(36) -- example: ['891365b3-6b62-4e92-b683-1694678b9e74', '79a4557d-991b-4838-ba41-9c2f436cc6f6', '943ef747-4626-4379-b092-d1cc3be828f0'],
    architecture VARCHAR(255),
    root_device_name VARCHAR(255) -- example: ['/dev/vda', '/dev/vda', '/dev/vda'],
    access_ip_v4 VARCHAR(39),
    access_ip_v6 VARCHAR(39),
    config_drive VARCHAR(255),
    task_state VARCHAR(255),
    default_ephemeral_device VARCHAR(255),
    default_swap_device VARCHAR(255),
    progress INT -- example: ['0', '0', '0'],
    auto_disk_config TINYINT(1) -- example: ['0', '0', '0'],
    shutdown_terminate TINYINT(1) -- example: ['0', '0', '0'],
    disable_terminate TINYINT(1) -- example: ['0', '0', '0'],
    root_gb INT -- example: ['64', '16', '16'],
    ephemeral_gb INT -- example: ['0', '0', '0'],
    cell_name VARCHAR(255),
    node VARCHAR(255),
    deleted INT -- example: ['0', '0', '0'],
    locked_by ENUM('OWNER','ADMIN'),
    cleaned INT -- example: ['1', '1', '1'],
    ephemeral_key_uuid VARCHAR(36)
);

CREATE TABLE instance_info_caches (
    created_at DATETIME -- example: ['2014-10-15T01:44:40', '2014-10-19T01:38:54', '2014-10-19T02:02:46'],
    updated_at DATETIME -- example: ['2023-01-31T14:45:53', '2023-01-31T13:49:15', '2023-01-31T14:50:06'],
    deleted_at DATETIME,
    id INT -- example: ['509070', '509284', '509285'],
    network_info TEXT,
    instance_uuid VARCHAR(36) -- example: ['891365b3-6b62-4e92-b683-1694678b9e74', '79a4557d-991b-4838-ba41-9c2f436cc6f6', '943ef747-4626-4379-b092-d1cc3be828f0'],
    deleted INT -- example: ['0', '0', '0']
);

-- Schema Mapping (question concepts to columns):
-- 'instance (not deleted)' -> instances.deleted
-- 'instance UUID' -> instances.uuid
-- 'instance display name' -> instances.display_name
-- 'system metadata key image_image_type' -> instance_system_metadata.key
-- 'metadata value for that key' -> instance_system_metadata.value
-- 'network-info cache records' -> instance_info_caches.id
-- 'total memory (in MB) across cache records' -> instances.memory_mb

You should use the provided mapping to determine which columns and tables should be used in the SQL statement.

-- Join Keys (how tables connect):
-- instance_system_metadata.instance_uuid = instances.uuid
-- instances.uuid = instance_info_caches.instance_uuid

You should use the provided join keys to determine how to connect the tables in the SQL statement.

Example question:
For each instance on host `blaze8-12` that is not deleted and has system metadata with the key `image_image_type`, return the instance UUID, the instance display name, the metadata value for that key, the number of network-info cache records for the instance, and the total memory (in MB) summed across those cache records.

Example SQL (MySQL dialect):
```sql
SELECT i.uuid AS instance_uuid, i.display_name, ism.value AS image_type_value, COUNT(iic.id) AS cache_record_count, SUM(i.memory_mb) AS total_memory_mb FROM instances i JOIN instance_system_metadata ism ON ism.instance_uuid = i.uuid JOIN instance_info_caches iic ON iic.instance_uuid = i.uuid WHERE i.deleted = 0 AND ism.key = 'image_image_type' AND i.host = 'blaze8-12' GROUP BY i.uuid, i.display_name, ism.value;
```
'''
        if option == 2:
            import re
            txt = re.sub(r'-- Schema Mapping \(question concepts to columns\):.*SQL statement\.\s+', '', txt, flags=re.DOTALL)
            txt = re.sub(r'-- Join Keys \(how tables connect\):.*SQL statement\.\s+', '', txt, flags=re.DOTALL)
        return txt

    def get_dw_one_shot_example(self, option=2):
        txt = r'''[One-shot Example]
Database ID: dw

[Database description]
CREATE TABLE COURSE_CATALOG_SUBJECT_OFFERED (
    ACADEMIC_YEAR VARCHAR2 -- example: ['2021', '2024', '2023'],
    TERM_CODE VARCHAR2 -- example: ['2021SP', '2019SP', '2014SP'],
    SUBJECT_ID VARCHAR2 -- example: ['18.03', '5.12', '5.111'],
    SUBJECT_CODE VARCHAR2 -- example: ['15', '6', '2'],
    SUBJECT_NUMBER VARCHAR2 -- example: ['THG', 'UR', '03'],
    SOURCE_SUBJECT_ID VARCHAR2 -- example: ['18.03', '6.046', '5.12'],
    PRINT_SUBJECT_ID VARCHAR2 -- example: ['18.03', '5.12', '5.111'],
    IS_PRINTED_IN_BULLETIN VARCHAR2 -- example: ['Y', 'N', 'S'],
    DEPARTMENT_CODE VARCHAR2 -- example: ['15', '6', '2'],
    DEPARTMENT_NAME VARCHAR2 -- example: ['Management', 'Electrical Eng & Computer Sci', 'Mechanical Engineering'],
    EFFECTIVE_TERM_CODE VARCHAR2 -- example: ['2016FA', '2018FA', '2017FA'],
    SUBJECT_SHORT_TITLE VARCHAR2 -- example: ['Undergraduate Research', 'Special Seminar in Management', 'Calculus'],
    SUBJECT_TITLE VARCHAR2 -- example: ['Undergraduate Research', 'Special Seminar in Management', 'Graduate Thesis'],
    IS_VARIABLE_UNITS VARCHAR2 -- example: ['N', 'Y'],
    LECTURE_UNITS NUMBER -- example: ['3', '0', '4'],
    LAB_UNITS NUMBER -- example: ['0', '2', '3'],
    PREPARATION_UNITS NUMBER -- example: ['9', '0', '8'],
    TOTAL_UNITS NUMBER -- example: ['12', '0', '6'],
    DESIGN_UNITS NUMBER -- example: ['0', '4', '12'],
    GRADE_TYPE VARCHAR2 -- example: ['L', 'P'],
    GRADE_TYPE_DESC VARCHAR2 -- example: ['Letter graded', 'P/D/F'],
    GRADE_RULE VARCHAR2 -- example: ['N', 'R', 'J'],
    GRADE_RULE_DESC VARCHAR2 -- example: ['Not repeatable for credit', 'Can be repeated for credit', 'Continuing and Repeatable'],
    HGN_CODE VARCHAR2 -- example: ['U', 'G', 'H'],
    HGN_DESC VARCHAR2 -- example: ['Undergraduate', 'Graduate', 'High Graduate'],
    HGN_EXCEPT VARCHAR2 -- example: ['(H except 18)', '(H except XVIII)', '(H except 2, 6, 8, 12, 13, 16, 18, 22)'],
    GIR_ATTRIBUTE VARCHAR2 -- example: ['HE', 'REST', 'LAB'],
    GIR_ATTRIBUTE_DESC VARCHAR2 -- example: ['HASS Elective', 'Rest Elec in Sci & Tech', 'Institute Lab'],
    COMM_REQ_ATTRIBUTE VARCHAR2 -- example: ['CIM', 'CIH', 'CIHW'],
    COMM_REQ_ATTRIBUTE_DESC VARCHAR2 -- example: ['Communication Intensive Major', 'Communication Intensive HASS', 'Communication Intensive Writing'],
    TUITION_ATTRIBUTE VARCHAR2 -- example: ['RESH', 'NTRN', 'COOP'],
    TUITION_ATTRIBUTE_DESC VARCHAR2 -- example: ['Pre-thesis Research Subject', 'Internship', 'Co-op Subject'],
    WRITE_REQ_ATTRIBUTE VARCHAR2 -- example: ['WRT2', 'WRT1'],
    WRITE_REQ_ATTRIBUTE_DESC VARCHAR2 -- example: ['Writing Requirement, Phase II', 'Writing Requirement, Phase I'],
    SUPERVISOR_ATTRIBUTE VARCHAR2 -- example: ['UROP', 'INDP', 'THG'],
    SUPERVISOR_ATTRIBUTE_DESC VARCHAR2 -- example: ['UROP subject', 'Independent Study', 'Grad Thesis'],
    PREREQUISITES VARCHAR2 -- example: ['Permission of instructor', 'GIR:CAL1', 'GIR:CAL2'],
    SUBJECT_DESCRIPTION VARCHAR2 -- example: ['Group study of current topics related to management not otherwise included in curriculum.', 'Covers subject matter not offered in the regular curriculum. Consult department to learn of offerings for a particular term.', 'Supplementary work on individual or group basis. Registration subject to prior arrangement for subject matter and supervision by staff.'],
    JOINT_SUBJECTS VARCHAR2 -- example: ['18.410J', '18.062J', '6.046J'],
    SCHOOL_WIDE_ELECTIVES VARCHAR2 -- example: ['1.EPW, 2.EPW, 3.EPW, 6.EPW, 10.EPW, 16.EPW, 20.EPW, 22.EPW', '2.96, 6.930, 10.806, 16.653', '1.EPE, 2.EPE, 3.EPE, 6.EPE, 8.EPE, 10.EPE, 15.EPE, 16.EPE, 20.EPE, 22.EPE'],
    MEETS_WITH_SUBJECTS VARCHAR2 -- example: ['6.431', '2.791J, 6.021J, 20.370J', '1.001'],
    EQUIVALENT_SUBJECTS VARCHAR2 -- example: ['18.700', '18.034', '5.111, 5.112, CC.5111, ES.5111, ES.5112'],
    IS_OFFERED_THIS_YEAR VARCHAR2 -- example: ['Y', 'N'],
    IS_OFFERED_FALL_TERM VARCHAR2 -- example: ['Y', 'N'],
    IS_OFFERED_IAP VARCHAR2 -- example: ['N', 'Y'],
    IS_OFFERED_SPRING_TERM VARCHAR2 -- example: ['Y', 'N'],
    IS_OFFERED_SUMMER_TERM VARCHAR2 -- example: ['N', 'Y'],
    FALL_INSTRUCTORS VARCHAR2 -- example: ['Staff', 'F. E. Palmer', 'K. Schultz'],
    SPRING_INSTRUCTORS VARCHAR2 -- example: ['Staff', 'F. E. Palmer', 'K. Schultz'],
    STATUS_CHANGE VARCHAR2 -- example: ['New subject', 'New joint child', '(2012: Removed subject)'],
    LAST_ACTIVITY_DATE DATE -- example: ['27-OCT-22', '26-OCT-23', '28-OCT-21'],
    WAREHOUSE_LOAD_DATE DATE -- example: ['19-DEC-24'],
    MASTER_SUBJECT_ID VARCHAR2 -- example: ['10.26', '10.01', '5.111'],
    HASS_ATTRIBUTE VARCHAR2 -- example: ['HH', 'HS', 'HA'],
    HASS_ATTRIBUTE_DESC VARCHAR2 -- example: ['HASS Humanities', 'HASS Social Sciences', 'HASS Arts'],
    TERM_DURATION VARCHAR2 -- example: ['Full Term Subject', 'First Half Term Subject', 'Second Half Term Subject'],
    GLOBAL_REGIONS VARCHAR2 -- example: ['Global (all regions)', 'Asia', 'Europe'],
    GLOBAL_COUNTRIES VARCHAR2 -- example: ['China', 'France', 'Japan'],
    ON_LINE_PAGE_NUMBER VARCHAR2 -- example: ['http://student.mit.edu/catalog/m6a.html', 'http://student.mit.edu/catalog/m15b.html', 'http://student.mit.edu/catalog/m15c.html'],
    SECTION_ID VARCHAR2 -- example: ['000', 'L01', 'R01'],
    IS_MASTER_SECTION VARCHAR2 -- example: ['N', 'Y'],
    IS_LECTURE_SECTION VARCHAR2 -- example: ['N', 'Y'],
    IS_LAB_SECTION VARCHAR2 -- example: ['N', 'Y'],
    IS_RECITATION_SECTION VARCHAR2 -- example: ['N', 'Y'],
    IS_DESIGN_SECTION VARCHAR2 -- example: ['N', 'Y'],
    RESPONSIBLE_FACULTY_NAME VARCHAR2 -- example: ['Petty, Mustafa', 'Pugh, Elin', 'Mooney, Francesco'],
    RESPONSIBLE_FACULTY_MIT_ID VARCHAR2 -- example: ['920324608.0', '916610219.0', '925785734.0'],
    MEET_TIME VARCHAR2 -- example: ['*TO BE ARRANGED', 'TR1-2.30', 'TR11-12.30'],
    MEET_PLACE VARCHAR2 -- example: ['VIRTUAL', '1-337A', 'E51-385D']
);

CREATE TABLE FCLT_ROOMS (
    FCLT_ROOM_KEY VARCHAR2 -- example: ['E62-420', 'E52-420', 'W61-055'],
    BUILDING_ROOM VARCHAR2 -- example: ['E62-420', 'E52-420', 'W61-055'],
    FCLT_BUILDING_KEY VARCHAR2 -- example: ['46', '32', 'E37'],
    FLOOR VARCHAR2 -- example: ['1', '2', '3'],
    FCLT_FLOOR_KEY VARCHAR2 -- example: ['46-6', '46-4', '46-5'],
    ROOM VARCHAR2 -- example: ['187', '121', '393'],
    SPACE_ID VARCHAR2 -- example: ['E62-4-420', '38-3-393', 'W20-1-136'],
    FCLT_MAJOR_USE_KEY VARCHAR2 -- example: ['108', '102', '109'],
    MAJOR_USE_DESC VARCHAR2 -- example: ['OFFICES', 'CIRCULAT', 'RESIDENT'],
    FCLT_USE_KEY VARCHAR2 -- example: ['158', '159', '169'],
    USE_DESC VARCHAR2,
    FCLT_MINOR_USE_KEY VARCHAR2,
    MINOR_USE_DESC VARCHAR2,
    FCLT_ORGANIZATION_KEY VARCHAR2 -- example: ['149', '236', '235'],
    ORGANIZATION_NAME VARCHAR2 -- example: ['DOF', 'RESIDE', 'RESDOF'],
    FCLT_MINOR_ORGANIZATION_KEY VARCHAR2,
    MINOR_ORGANIZATION VARCHAR2,
    AREA NUMBER -- example: ['42.0', '209.21', '321.12'],
    ROOM_FULL_NAME VARCHAR2 -- example: ['MENS  LOCKER', 'WOMENS LOCKER', 'MENS TEAM ROOM'],
    DEPT_CODE VARCHAR2 -- example: ['93700.0', '93300.0', '93400.0'],
    ACCESS_LEVEL VARCHAR2 -- example: ['2', '1', '3'],
    LATITUDE_WGS NUMBER,
    LONGITUDE_WGS NUMBER,
    NORTHING_SPCS NUMBER,
    EASTING_SPCS NUMBER,
    WAREHOUSE_LOAD_DATE DATE -- example: ['19-DEC-24']
);

CREATE TABLE FCLT_BUILDING_ADDRESS (
    FCLT_BUILDING_ADDRESS_KEY VARCHAR2 -- example: ['66-STREET', 'W84-PARCL2', 'W85-E911_1'],
    FCLT_BUILDING_KEY DATE -- example: ['W70', 'W4', 'W53'],
    BUILDING_NUMBER VARCHAR2 -- example: ['W70', 'W4', 'W53'],
    ADDRESS_PURPOSE VARCHAR2 -- example: ['STREET', 'E911_1', 'MAIL'],
    ADDRESS_CITY_ID VARCHAR2 -- example: ['708.0', '44.0', '25343.0'],
    IS_E911_ADDRESS VARCHAR2,
    STREET_NUMBER VARCHAR2 -- example: ['77', 'MEM106', '21'],
    STREET_NUMBER_SUFFIX VARCHAR2 -- example: ['R'],
    PRE_DIRECTIONAL VARCHAR2,
    STREET_NAME VARCHAR2 -- example: ['MASSACHUSETTS', 'MEMORIAL', 'VASSAR'],
    STREET_SUFFIX VARCHAR2 -- example: ['ST', 'AVE', 'DR'],
    POST_DIRECTIONAL VARCHAR2 -- example: ['(Rear)', 'NE', 'NW'],
    CITY VARCHAR2 -- example: ['CAMBRIDGE', 'MIDDLETON', 'WESTFORD'],
    STATE VARCHAR2 -- example: ['MA', 'DC'],
    POSTAL_CODE VARCHAR2 -- example: ['2139', '2142', '1949'],
    WAREHOUSE_LOAD_DATE VARCHAR2 -- example: ['19-DEC-24']
);

CREATE TABLE FCLT_BUILDING (
    FCLT_BUILDING_KEY VARCHAR2 -- example: ['2', 'W51C', '11'],
    BUILDING_NUMBER VARCHAR2 -- example: ['2', 'W51C', '11'],
    PARENT_BUILDING_NUMBER VARCHAR2 -- example: ['W61', '14', '62'],
    PARENT_BUILDING_NAME VARCHAR2 -- example: ['MACGREGOR HOUSE', 'HAYDEN MEMORIAL LIBRARY', 'ALUMNI HOUSES: MUNROE HAYDEN WOOD'],
    PARENT_BUILDING_NAME_LONG VARCHAR2 -- example: ['Frank S MacGregor House', 'Charles Hayden Memorial Library', 'Alumni Houses: Munroe Hayden Wood'],
    BUILDING_NAME_LONG VARCHAR2 -- example: ['WALLACE ASTROPHYSICAL OBSERVATORY', 'J B Carr Indoor Tennis Facility (Office)', 'THE SIMONS BUILDING'],
    EXT_GROSS_AREA NUMBER -- example: ['0.0', '100.38', '12634.2'],
    ASSIGNABLE_AREA NUMBER -- example: ['0.0', '87.21', '7417.78'],
    NON_ASSIGNABLE_AREA NUMBER -- example: ['0.0', '454.55', '904.31'],
    SITE VARCHAR2 -- example: ['MIT', 'BATES', 'HAY'],
    CAMPUS_SECTOR VARCHAR2 -- example: ['WEST', 'MAIN GROUP', 'OFFCAMPUS'],
    ACCESS_LEVEL_CODE NUMBER -- example: ['2', '1', '0'],
    ACCESS_LEVEL_NAME VARCHAR2 -- example: ['2', '1', '0'],
    BUILDING_TYPE VARCHAR2 -- example: ['ACADEMIC', 'SERVICE', 'RESIDENT'],
    OWNERSHIP_TYPE VARCHAR2 -- example: ['OWNED', 'LEASED'],
    BUILDING_USE VARCHAR2 -- example: ['AER', 'DHOA', 'OTH'],
    OCCUPANCY_CLASS VARCHAR2 -- example: ['(NULL)', 'UGB', 'UGR2'],
    BUILDING_HEIGHT VARCHAR2 -- example: ['0.0', '31.1', '75.0'],
    COST_CENTER_CODE VARCHAR2 -- example: ['1876000.0', '1348000.0', '1810700.0'],
    COST_COLLECTOR_KEY VARCHAR2 -- example: ['1876000.0', '1348000.0', '1810700.0'],
    LATITUDE_WGS NUMBER -- example: ['42.35881273', '42.36107393', '42.35545034'],
    LONGITUDE_WGS NUMBER -- example: ['-71.09018269', '-71.0923498', '-71.10315273'],
    EASTING_X_SPCS NUMBER -- example: ['766931.08927', '766341.434677', '763431.304576999'],
    NORTHING_Y_SPCS NUMBER -- example: ['2956046.58619999', '2956867.80465999', '2954804.68813'],
    BUILDING_SORT VARCHAR2 -- example: ['02', 'W51C', '11'],
    BUILDING_NAMED_FOR VARCHAR2 -- example: ['-', 'CHARLES HAYDEN', 'R. C. MACLAURIN'],
    BUILDING_NAME VARCHAR2 -- example: ['WALLACE ASTROPHYSICAL OBSERVATORY', 'BUILDING W92', 'HOMBERG BUILDING'],
    DATE_BUILT VARCHAR2 -- example: ['07/01/1913', '01/01/1962', '12/31/1915'],
    DATE_ACQUIRED VARCHAR2 -- example: ['07/01/1963', '07/01/2016', '02/01/2016'],
    DATE_OCCUPIED VARCHAR2 -- example: ['12/31/1916', '08/01/1963', '08/05/2024'],
    WAREHOUSE_LOAD_DATE DATE -- example: ['19-DEC-24'],
    NUM_OF_ROOMS NUMBER -- example: ['1', '10', '0']
);

CREATE TABLE BUILDINGS (
    BUILDING_KEY VARCHAR2 -- example: ['NW32', 'E18', '46'],
    BUILDING_NUMBER VARCHAR2 -- example: ['NW32', 'E18', '46'],
    BUILDING_NAME VARCHAR2 -- example: ['WALLACE ASTROPHYSICAL OBSERVATORY', 'J B Carr Indoor Tennis Facility (Office)', 'BUILDING NW32'],
    BUILDING_STREET_ADDRESS VARCHAR2 -- example: ['21 MANNING AVE', '244  WOOD ST', '410 MEMORIAL DR (REAR)'],
    BUILDING_MAILING_ADDRESS VARCHAR2,
    BLDG_GROSS_SQUARE_FOOTAGE NUMBER -- example: ['0.0', '100.38', '9320.91'],
    BLDG_ASSIGNABLE_SQUARE_FOOTAGE NUMBER -- example: ['0.0', '87.21', '7417.78'],
    BUILDING_COUNTER NUMBER -- example: ['1'],
    WAREHOUSE_LOAD_DATE DATE -- example: ['19-DEC-24']
);
-- Schema Mapping (question concepts to columns):
-- 'full room names' -> FCLT_ROOMS.ROOM_FULL_NAME
-- 'building names' -> BUILDINGS.BUILDING_NAME
-- 'street addresses' -> BUILDINGS.BUILDING_STREET_ADDRESS, FCLT_BUILDING_ADDRESS.ADDRESS_PURPOSE
-- 'cities' -> FCLT_BUILDING_ADDRESS.CITY
-- 'states' -> FCLT_BUILDING_ADDRESS.STATE
-- 'postal codes' -> FCLT_BUILDING_ADDRESS.POSTAL_CODE
-- 'building heights' -> FCLT_BUILDING.BUILDING_HEIGHT
-- 'Computer Science' -> COURSE_CATALOG_SUBJECT_OFFERED.DEPARTMENT_NAME


You should use the provided mapping to determine which columns and tables should be used in the SQL statement.
-- Join Keys (how tables connect):
-- COURSE_CATALOG_SUBJECT_OFFERED.MEET_PLACE = FCLT_ROOMS.FCLT_ROOM_KEY
-- FCLT_BUILDING_ADDRESS.FCLT_BUILDING_KEY = FCLT_ROOMS.FCLT_BUILDING_KEY
-- FCLT_BUILDING.FCLT_BUILDING_KEY = FCLT_ROOMS.FCLT_BUILDING_KEY
-- BUILDINGS.BUILDING_KEY = FCLT_ROOMS.FCLT_BUILDING_KEY


You should use the provided join keys to determine how to connect the tables in the SQL statement.

-- External Knowledge (database-wide):
"street address" is predicated by "TABLE.ADDRESS_PURPOSE = 'STREET'"
"Computer Science" is predicated by "TABLE.DEPARTMENT_NAME = 'Electrical Eng & Computer Sci'"
You should use the external knowledge to help determine which tables and columns to use in the SQL statement as well as constructing the SQL statement.

Example question:
List the unique full room names, their corresponding building names, street addresses, cities, states, postal codes, and building heights for rooms associated with subjects that Computer Science students can enroll in.

Example SQL (MySQL dialect):
```sql
SELECT DISTINCT fr.ROOM_FULL_NAME, b.BUILDING_NAME, b.BUILDING_STREET_ADDRESS, fba.CITY, fba.STATE, fba.POSTAL_CODE, fb.BUILDING_HEIGHT FROM COURSE_CATALOG_SUBJECT_OFFERED JOIN FCLT_ROOMS fr ON MEET_PLACE = FCLT_ROOM_KEY JOIN FCLT_BUILDING_ADDRESS fba ON fba.FCLT_BUILDING_KEY = fr.FCLT_BUILDING_KEY JOIN FCLT_BUILDING fb ON fb.FCLT_BUILDING_KEY = fr.FCLT_BUILDING_KEY JOIN BUILDINGS b ON b.BUILDING_KEY = fr.FCLT_BUILDING_KEY WHERE DEPARTMENT_NAME = 'Electrical Eng & Computer Sci' AND fba.ADDRESS_PURPOSE = 'STREET';
```
```
'''
        import re
        if option == 1:
            txt = re.sub(r'-- Schema Mapping \(question concepts to columns\):.*-- Join Keys \(how tables connect\):', '-- Join Keys (how tables connect):', txt, flags=re.DOTALL)
            txt = re.sub(r'-- Join Keys \(how tables connect\):.*-- External Knowledge \(database-wide\):', '-- External Knowledge (database-wide):', txt, flags=re.DOTALL)
            txt = re.sub(r'-- External Knowledge \(database-wide\):.*Example question:', 'Example question:', txt, flags=re.DOTALL)
        
        elif option < 3:
             txt = re.sub(r'-- External Knowledge \(database-wide\):.*Example question:', 'Example question:', txt, flags=re.DOTALL)

        return txt

    def get_self_refine_prompt(self, table_info, task, pre_info, question, api, format_csv, table_struct, omnisql_format_pth=None, db_id="dw", option=2):
        if omnisql_format_pth:
            if task == "lite":
                return omni_sql_input_prompt_template.format(
                    db_engine = "SQLite",
                    db_details = table_info,
                    question = question
                )
            elif task in ["BIRD", "spider"]:
                ce = "Some few-shot examples after column exploration may be helpful:\n" + pre_info if pre_info else ""
                return table_info + "\n" + ce
        refine_prompt = table_info + "\n"
        # refine_prompt += "Begin Exploring Related Columns\n" + response_pre_txt + "\nRefined SQLs and results:\n" + pre_info + "End Exploring Related Columns\n" if pre_info else ""
        refine_prompt += "Some few-shot examples after column exploration may be helpful:\n" + pre_info if pre_info else ""

        # Add one-shot example for Beaver (particularly helpful for option 4 with full context)
        if task == "beaver":
            refine_prompt += "\n" + self.get_beaver_one_shot_example(db_id=db_id, option=option) + "\n"

        refine_prompt += "Task: " + question + "\n"+f'\nPlease think step by step and answer only one complete SQL in {api} dialect in ```sql``` format.\n'
        refine_prompt += f'SQL usage example: {self.get_prompt_dialect_basic(api)}\n'
        refine_prompt += f"Follow the answer format like: {format_csv}.\n" if format_csv else ""
        # refine_prompt += "Here are some useful tips for answering:\n"
        
        # refine_prompt += self.get_prompt_dialect_list_all_tables(table_struct, api)
        # refine_prompt += self.get_prompt_fuzzy_query()

        # if api == "snowflake":
        #     refine_prompt += "When using ORDER BY xxx DESC, add NULLS LAST to exclude null records: ORDER BY xxx DESC NULLS LAST.\n"
        # refine_prompt += "When using ORDER BY, if there are duplicate values in the primary sort column, sort by an additional column as a secondary criterion.\n"
        
        # Specific:
        # refine_prompt += "When asked something without stating name or id, return both of them. e.g. Which products ...? The answer should include product_name and product_id.\n"
        # refine_prompt += "When asked percentage decrease, you should return a positive value. e.g. How many percentage points in 2021 decrease compared to ...? The answer should be a positive value indicating the decresed number. Try to use ABS().\n"
        # refine_prompt += "If asked two tables, you should reply with the last one instead of combining two tables. e.g. Identifying the top five states ... examine the state that ranks fourth overall and identify its top five counties. You should only answer top five counties.\n"
        # if api == "snowflake":
        #     refine_prompt += "Use ST_DISTANCE to calculate distance between two geographic points for more accurate answer.\n"
        # refine_prompt += self.get_prompt_decimal_places()
        
        return refine_prompt

    def get_self_consistency_prompt(self, task, format_csv):
        self_consistency_prompt = f"Please check the answer again by reviewing task:\n {task}\n, reviewing Relevant Tables and Columns and Possible Conditions and then give the final SQL query. Don't output other queries. If you think the answer is right, just output the current SQL.\n" 
        self_consistency_prompt += self.get_prompt_decimal_places()
        self_consistency_prompt += f"The answer format should be like: {format_csv}\n" if format_csv else ""

        return self_consistency_prompt
