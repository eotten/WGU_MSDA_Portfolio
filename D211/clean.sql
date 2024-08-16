-- Step 0: Clean the public.customer table
-- Update the churn column
UPDATE public.customer
SET churn = CASE
    WHEN churn = 'Yes' THEN TRUE
    WHEN churn = 'No' THEN FALSE
END;

-- Update the techie column
UPDATE public.customer
SET techie = CASE
    WHEN techie = 'Yes' THEN TRUE
    WHEN techie = 'No' THEN FALSE
END;

-- Update the port_modem column
UPDATE public.customer
SET port_modem = CASE
    WHEN port_modem = 'Yes' THEN TRUE
    WHEN port_modem = 'No' THEN FALSE
END;

-- Update the tablet column
UPDATE public.customer
SET tablet = CASE
    WHEN tablet = 'Yes' THEN TRUE
    WHEN tablet = 'No' THEN FALSE
END;


-- Step 1: Create the original_data table
CREATE TABLE original_data (
    id SERIAL PRIMARY KEY,
    "Label (Grouping)" TEXT,
    "United States!!Total!!Estimate" TEXT,
    "United States!!Total!!Margin of Error" TEXT,
    "United States!!Percent!!Estimate" TEXT,
    "United States!!Percent!!Margin of Error" TEXT,
    "United States!!Male!!Estimate" TEXT,
    "United States!!Male!!Margin of Error" TEXT,
    "United States!!Percent Male!!Estimate" TEXT,
    "United States!!Percent Male!!Margin of Error" TEXT,
    "United States!!Female!!Estimate" TEXT,
    "United States!!Female!!Margin of Error" TEXT,
    "United States!!Percent Female!!Estimate" TEXT,
    "United States!!Percent Female!!Margin of Error" TEXT
);

-- Step 2: Import data from A.csv
COPY original_data(
    "Label (Grouping)", 
    "United States!!Total!!Estimate", 
    "United States!!Total!!Margin of Error", 
    "United States!!Percent!!Estimate", 
    "United States!!Percent!!Margin of Error", 
    "United States!!Male!!Estimate", 
    "United States!!Male!!Margin of Error", 
    "United States!!Percent Male!!Estimate", 
    "United States!!Percent Male!!Margin of Error", 
    "United States!!Female!!Estimate", 
    "United States!!Female!!Margin of Error", 
    "United States!!Percent Female!!Estimate", 
    "United States!!Percent Female!!Margin of Error"
)
FROM 'C:/Users/Eric/Dropbox/University/WGU MSDA/D211/A.csv' 
DELIMITER ',' 
CSV HEADER;

-- Rename columns one at a time
ALTER TABLE original_data RENAME COLUMN "Label (Grouping)" TO label;
ALTER TABLE original_data RENAME COLUMN "United States!!Total!!Estimate" TO total;
ALTER TABLE original_data RENAME COLUMN "United States!!Total!!Margin of Error" TO total_me;
ALTER TABLE original_data RENAME COLUMN "United States!!Percent!!Estimate" TO percent;
ALTER TABLE original_data RENAME COLUMN "United States!!Percent!!Margin of Error" TO percent_me;
ALTER TABLE original_data RENAME COLUMN "United States!!Male!!Estimate" TO male;
ALTER TABLE original_data RENAME COLUMN "United States!!Male!!Margin of Error" TO male_me;
ALTER TABLE original_data RENAME COLUMN "United States!!Percent Male!!Estimate" TO percent_male;
ALTER TABLE original_data RENAME COLUMN "United States!!Percent Male!!Margin of Error" TO percent_male_me;
ALTER TABLE original_data RENAME COLUMN "United States!!Female!!Estimate" TO female;
ALTER TABLE original_data RENAME COLUMN "United States!!Female!!Margin of Error" TO female_me;
ALTER TABLE original_data RENAME COLUMN "United States!!Percent Female!!Estimate" TO percent_female;
ALTER TABLE original_data RENAME COLUMN "United States!!Percent Female!!Margin of Error" TO percent_female_me;

-- Clean the labels by removing non-standard whitespace characters
UPDATE original_data
SET label = TRIM(' ' FROM label);

-- Step 4: Create subtables for each category

-- Subtable for age
CREATE TABLE census_age_data AS
SELECT *
FROM original_data
WHERE id BETWEEN 3 AND 20;

-- Subtable for age categories
CREATE TABLE census_age_categories AS
SELECT *
FROM original_data
WHERE id BETWEEN 22 AND 34;

-- Subtable for summary
CREATE TABLE census_age_summary AS
SELECT *
FROM original_data
WHERE id BETWEEN 35 AND 40;

-- Subtable for percentages
CREATE TABLE census_age_percentages AS
SELECT *
FROM original_data
WHERE id BETWEEN 41 AND 43;

-- Step 5: Export subtables to CSV files

COPY census_age_data TO 'C:/Users/Eric/Dropbox/University/WGU MSDA/D211/cleaned/census_age_data.csv' DELIMITER ',' CSV HEADER;
COPY census_age_categories TO 'C:/Users/Eric/Dropbox/University/WGU MSDA/D211/cleaned/census_age_categories.csv' DELIMITER ',' CSV HEADER;
COPY census_age_summary TO 'C:/Users/Eric/Dropbox/University/WGU MSDA/D211/cleaned/census_age_summary.csv' DELIMITER ',' CSV HEADER;
COPY census_age_percentages TO 'C:/Users/Eric/Dropbox/University/WGU MSDA/D211/cleaned/census_age_percentages.csv' DELIMITER ',' CSV HEADER;

SELECT
CASE
	WHEN age BETWEEN 0 AND 4 THEN 'Under 5 years'
	WHEN age BETWEEN 5 AND 9 THEN '5 to 9 years'
	WHEN age BETWEEN 10 AND 14 THEN '10 to 14 years'
	WHEN age BETWEEN 15 AND 19 THEN '15 to 19 years'
	WHEN age BETWEEN 20 AND 24 THEN '20 to 24 years'
	WHEN age BETWEEN 25 AND 29 THEN '25 to 29 years'
	WHEN age BETWEEN 30 AND 34 THEN '30 to 34 years'
	WHEN age BETWEEN 35 AND 39 THEN '35 to 39 years'
	WHEN age BETWEEN 40 AND 44 THEN '40 to 44 years'
	WHEN age BETWEEN 45 AND 49 THEN '45 to 49 years'
	WHEN age BETWEEN 50 AND 54 THEN '50 to 54 years'
	WHEN age BETWEEN 55 AND 59 THEN '55 to 59 years'
	WHEN age BETWEEN 60 AND 64 THEN '60 to 64 years'
	WHEN age BETWEEN 65 AND 69 THEN '65 to 69 years'
	WHEN age BETWEEN 70 AND 74 THEN '70 to 74 years'
	WHEN age BETWEEN 75 AND 79 THEN '75 to 79 years'
	WHEN age BETWEEN 80 AND 84 THEN '80 to 84 years'
	WHEN age >= 85 THEN '85 years and over'
END AS age_group,
COUNT(*) * 100.0 / (SELECT COUNT(*) FROM public.customer) AS customer_percentage
INTO customer_age_groups
FROM public.customer
GROUP BY age_group;

COPY customer_age_groups TO 'C:/Users/Eric/Dropbox/University/WGU MSDA/D211/cleaned/customer_age_groups.csv' DELIMITER ',' CSV HEADER;
COPY public.customer TO 'C:/Users/Eric/Dropbox/University/WGU MSDA/D211/cleaned/customer.csv' DELIMITER ',' CSV HEADER;