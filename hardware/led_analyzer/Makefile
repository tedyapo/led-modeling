all: light_sensor

EXTENSIONS = GTL GBL GTS GBS GTO GBO GKO XLN

light_sensor: BASE_NAME=light_sensor
light_sensor:
	rm -f light_sensor.zip
	for ext in ${EXTENSIONS}; do\
    zip "${BASE_NAME}.zip" "${BASE_NAME}.$${ext}";\
    mv "${BASE_NAME}.$${ext}" gerbers;\
  done

clean:
	rm -f *~ *.b#[1-9] *.s#[1-9] *.zip *.dri *.gpi *.GML *.GTP

