package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
)

type Stud struct {
	Arithmancy                float64
	Astronomy                 float64
	Herbology                 float64
	DefenseAgainsttheDarkArts float64
	Divination                float64
	MuggleStudies             float64
	AncientRunes              float64
	HistoryofMagic            float64
	Transfiguration           float64
	Potions                   float64
	CareofMagicalCreatures    float64
	Charms                    float64
	Flying                    float64
}

const (
	lenFeatures            = 19
	indexNumericalFeatures = 6
	numericalFeatures      = lenFeatures - indexNumericalFeatures
)

var (
	studs    []Stud
	features [numericalFeatures]string
	count    [numericalFeatures]int
	mean     [numericalFeatures]float64
	std      [numericalFeatures]float64
	min      [numericalFeatures]float64
	q1       [numericalFeatures]float64
	q2       [numericalFeatures]float64
	q3       [numericalFeatures]float64
	max      [numericalFeatures]float64
)

func initMins() {
	for i := 0; i < numericalFeatures; i++ {
		min[i] = math.MaxUint64
	}
}

func main() {
	csvFile, _ := os.Open("../../assets/dataset_train.csv")
	reader := csv.NewReader(bufio.NewReader(csvFile))
	line, err := reader.Read()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(len(line))
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
		var f [numericalFeatures]float64
		for i := indexNumericalFeatures; i < lenFeatures; i++ {
			j := i - indexNumericalFeatures
			v, _ := strconv.ParseFloat(line[i], 64)
			// save value for building []Stud
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
		// studs := append(studs, Stud{
		// 	Arithmancy:                f[0],
		// 	Astronomy:                 f[1],
		// 	Herbology:                 f[2],
		// 	DefenseAgainsttheDarkArts: f[3],
		// 	Divination:                f[4],
		// 	MuggleStudies:             f[5],
		// 	AncientRunes:              f[6],
		// 	HistoryofMagic:            f[7],
		// 	Transfiguration:           f[8],
		// 	Potions:                   f[9],
		// 	CareofMagicalCreatures:    f[10],
		// 	Charms:                    f[11],
		// 	Flying:                    f[12],
		// },
		// )
		// get count
		count[0]++
	}
	for i := range count {
		count[i] = count[0]
	}
	// get mean
	for i := range mean {
		mean[i] /= float64(count[0])
	}
	fmt.Println("features", features)
	fmt.Println("count", count)
	fmt.Println("mean", mean)
	fmt.Println("min", min)
	fmt.Println("max", max)
}
