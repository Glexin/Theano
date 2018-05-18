if [ -f scan2.py ] && [ -f scan_std.py ]; then
	echo "error -> both file exists"
elif [ ! -f scan2.py ] && [ -f scan_std.py ]; then
	echo "scan.py scan2.py -> scan_std.py"
	echo "MY_scan -> old_scan"
	mv scan.py scan2.py
	mv scan_std.py scan.py
elif [ -f scan2.py ] && [ ! -f scan_std.py ]; then
	echo "scan.py scan_std.py -> scan2.py"
	echo "old_scan -> MY_scan"
	mv scan.py scan_std.py
	mv scan2.py scan.py
else
	echo "error -> both file not exists"
fi
