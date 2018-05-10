package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"math/big"
	"os"
	"reflect"
	"sort"
	"strconv"
	"text/tabwriter"
)

// Stud add or remove a numerical feature (eg. a mark) in the struct fields"
type Stud struct {
	F0  float64
	F1  float64
	F2  float64
	F3  float64
	F4  float64
	F5  float64
	F6  float64
	F7  float64
	F8  float64
	F9  float64
	F10 float64
	F11 float64
	F12 float64
}

// Modify these values according to the dataset and the structure Stud
const (
	filePath               = "assets/dataset_train.csv"
	lenFeatures            = 19
	indexNumericalFeatures = 6
	numericalFeatures      = lenFeatures - indexNumericalFeatures
)

var (
	studs    []Stud
	features [numericalFeatures]string
	count    int
	mean     [numericalFeatures]float64
	std      [numericalFeatures]float64
	min      [numericalFeatures]float64
	q1       [numericalFeatures]float64
	q2       [numericalFeatures]float64
	q3       [numericalFeatures]float64
	max      [numericalFeatures]float64
)

func floatToString(f float64) string {
	return big.NewFloat(f).Text('f', 2)
}

func print() {
	var (
		strFeatures = ""
		strCount    = "Count"
		strMean     = "Mean"
		strStd      = "Std"
		strMin      = "Min"
		strQ1       = "25%"
		strQ2       = "50%"
		strQ3       = "75%"
		strMax      = "Max"
	)

	for i := range mean {
		strFeatures += "\t" + features[i]
		strCount += "\t" + floatToString(float64(count))
		strMean += "\t" + floatToString(mean[i])
		strStd += "\t" + floatToString(std[i])
		strMin += "\t" + floatToString(min[i])
		strQ1 += "\t" + floatToString(q1[i])
		strQ2 += "\t" + floatToString(q2[i])
		strQ3 += "\t" + floatToString(q3[i])
		strMax += "\t" + floatToString(max[i])
	}

	w := new(tabwriter.Writer)

	// Format in tab-separated columns with a tab stop of 8.
	w.Init(os.Stdout, 0, 8, 2, '\t', 0)
	fmt.Fprintln(w, strFeatures)
	fmt.Fprintln(w, strCount)
	fmt.Fprintln(w, strMean)
	fmt.Fprintln(w, strStd)
	fmt.Fprintln(w, strMin)
	fmt.Fprintln(w, strQ1)
	fmt.Fprintln(w, strQ2)
	fmt.Fprintln(w, strQ3)
	fmt.Fprintln(w, strMax)
	fmt.Fprintln(w)
	w.Flush()
}

func getQuartiles() {
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].F0 < studs[j].F0
	})
	q1[0] = studs[count/4-1].F0
	q2[0] = (studs[count/2].F0 + studs[count/2-1].F0) / 2
	q3[0] = studs[count/4*3-1].F0
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].F1 < studs[j].F1
	})
	q1[1] = studs[count/4-1].F1
	q2[1] = (studs[count/2].F1 + studs[count/2-1].F1) / 2
	q3[1] = studs[count/4*3-1].F1
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].F2 < studs[j].F2
	})
	q1[2] = studs[count/4-1].F2
	q2[2] = (studs[count/2].F2 + studs[count/2-1].F2) / 2
	q3[2] = studs[count/4*3-1].F2
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].F3 < studs[j].F3
	})
	q1[3] = studs[count/4-1].F3
	q2[3] = (studs[count/2].F3 + studs[count/2-1].F3) / 2
	q3[3] = studs[count/4*3-1].F3
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].F4 < studs[j].F4
	})
	q1[4] = studs[count/4-1].F4
	q2[4] = (studs[count/2].F4 + studs[count/2-1].F4) / 2
	q3[4] = studs[count/4*3-1].F4
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].F5 < studs[j].F5
	})
	q1[5] = studs[count/4-1].F5
	q2[5] = (studs[count/2].F5 + studs[count/2-1].F5) / 2
	q3[5] = studs[count/4*3-1].F5
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].F6 < studs[j].F6
	})
	q1[6] = studs[count/4-1].F6
	q2[6] = (studs[count/2].F6 + studs[count/2-1].F6) / 2
	q3[6] = studs[count/4*3-1].F6
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].F7 < studs[j].F7
	})
	q1[7] = studs[count/4-1].F7
	q2[7] = (studs[count/2].F7 + studs[count/2-1].F7) / 2
	q3[7] = studs[count/4*3-1].F7
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].F8 < studs[j].F8
	})
	q1[8] = studs[count/4-1].F8
	q2[8] = (studs[count/2].F8 + studs[count/2-1].F8) / 2
	q3[8] = studs[count/4*3-1].F8
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].F9 < studs[j].F9
	})
	q1[9] = studs[count/4-1].F9
	q2[9] = (studs[count/2].F9 + studs[count/2-1].F9) / 2
	q3[9] = studs[count/4*3-1].F9
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].F10 < studs[j].F10
	})
	q1[10] = studs[count/4-1].F10
	q2[10] = (studs[count/2].F10 + studs[count/2-1].F10) / 2
	q3[10] = studs[count/4*3-1].F10
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].F11 < studs[j].F11
	})
	q1[11] = studs[count/4-1].F11
	q2[11] = (studs[count/2].F11 + studs[count/2-1].F11) / 2
	q3[11] = studs[count/4*3-1].F11
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].F12 < studs[j].F12
	})
	q1[12] = studs[count/4-1].F12
	q2[12] = (studs[count/2].F12 + studs[count/2-1].F12) / 2
	q3[12] = studs[count/4*3-1].F12
}

func getStd() {
	for _, s := range studs {
		stud := reflect.ValueOf(s)
		for i := 0; i < stud.NumField(); i++ {
			diff := stud.Field(i).Interface().(float64) - mean[i]
			std[i] += diff * diff
		}
	}
	for i := range std {
		std[i] /= float64(count)
		std[i] = math.Sqrt(std[i])
	}
}

func isFormatted(line []string) bool {
	for i := range line {
		if line[i] == "" {
			return false
		}
	}
	return true
}

func initMins() {
	for i := 0; i < numericalFeatures; i++ {
		min[i] = math.MaxUint64
	}
}

func main() {
	csvFile, _ := os.Open(filePath)
	defer csvFile.Close()
	reader := csv.NewReader(bufio.NewReader(csvFile))
	line, err := reader.Read()
	if err != nil {
		log.Fatal(err)
	}
	for i := indexNumericalFeatures; i < lenFeatures; i++ {
		j := i - indexNumericalFeatures
		features[j] = line[i]
	}
	initMins()
	for {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}
		if isFormatted(line) {
			var f [numericalFeatures]float64
			for i := indexNumericalFeatures; i < lenFeatures; i++ {
				j := i - indexNumericalFeatures
				v, _ := strconv.ParseFloat(line[i], 64)
				// save value in order to build []Stud
				f[j] = v
				// get min for each feature
				if v < min[j] {
					min[j] = v
				}
				// get max for each feature
				if v > max[j] {
					max[j] = v
				}
				mean[j] += v
			}
			studs = append(studs, Stud{
				F0:  f[0],
				F1:  f[1],
				F2:  f[2],
				F3:  f[3],
				F4:  f[4],
				F5:  f[5],
				F6:  f[6],
				F7:  f[7],
				F8:  f[8],
				F9:  f[9],
				F10: f[10],
				F11: f[11],
				F12: f[12],
			},
			)
			count++
		}
	}
	// fmt.Println(len(studs), cap(studs))
	// get mean
	for i := range mean {
		mean[i] /= float64(count)
	}
	getStd()
	getQuartiles()
	print()
}
