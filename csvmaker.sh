echo "author_name, time_sec, subject" > log.csv && git --no-pager log --pretty=format:%an,%at,%s >> log.csv
