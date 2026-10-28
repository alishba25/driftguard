# Data findings (Day 4)

1. Size: 48,842 rows, 15 columns
2. Numeric columns: 6 | Text (categorical) columns: 9 (8 features + the income target)
3. Missing values in: workclass, occupation, native-country
   (workclass 2,799 | occupation 2,809 | native-country 857)
4. Target: 23.9% earn >50K  -> the classes are imbalanced
5. Numeric ranges: age 17 to 90, hours-per-week 1 to 99
6. Most categories in one column: native-country (41 values)
7. Duplicate rows: 52

Questions this raised:
- occupation has 10 more missing values than workclass. Which rows have
  a workclass but no occupation, and why?
- capital-gain maxes out at exactly 99,999 and hours-per-week at 99. Are
  these real values or caps/placeholders?
- Most people have capital-gain and capital-loss of 0. Does that skew the
  scaling, and will drift tests on these columns behave oddly?
- native-country has 41 values, many probably rare. Will rare categories
  cause false drift alerts in small batches?
- 52 duplicate rows: real identical people, or repeated records?