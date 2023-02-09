#!/bin/sh

echo "Run baseline model"

echo "vital sign model"
python baselinemodel.py -s sign -m rf
echo "vital sign + Lab A model"
python baselinemodel.py -s lab_A -m rf
echo "vital sign + Lab B model"
python baselinemodel.py -s lab_B -m rf
echo "vital sign + lab C model"
python baselinemodel.py -s lab_C -m rf
echo "vital sign + lab D model"
python baselinemodel.py -s lab_D -m rf
echo "vital sign + lab E model"
python baselinemodel.py -s lab_E -m rf
echo "Lab F model"
python baselinemodel.py -s lab_F -m rf 
echo "vital sign + lab A + B model"
python baselinemodel.py -s lab_AB -m rf
echo "vital sign + lab A + B + C model"
python baselinemodel.py -s lab_ABC -m rf
echo "vital sign + lab A + B + C + D model"
python baselinemodel.py -s lab_ABCD -m rf
echo "full model"
python baselinemodel.py -s full -m rf
