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

// Modify these values according to the dataset and the structure Stud
const (
	filePath               = "../../assets/dataset_train.csv"
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
	// return strconv.FormatFloat(f, 'f', -1, 64)
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
	var isOdd int
	if count%2 == 0 {
		isOdd = 1
	}
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].Arithmancy < studs[j].Arithmancy
	})
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].Astronomy < studs[j].Astronomy
	})
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].Herbology < studs[j].Herbology
	})
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].DefenseAgainsttheDarkArts < studs[j].DefenseAgainsttheDarkArts
	})
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].Divination < studs[j].Divination
	})
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].MuggleStudies < studs[j].MuggleStudies
	})
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].AncientRunes < studs[j].AncientRunes
	})
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].HistoryofMagic < studs[j].HistoryofMagic
	})
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].Transfiguration < studs[j].Transfiguration
	})
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].Potions < studs[j].Potions
	})
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].CareofMagicalCreatures < studs[j].CareofMagicalCreatures
	})
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].Charms < studs[j].Charms
	})
	sort.Slice(studs, func(i, j int) bool {
		return studs[i].Flying < studs[j].Flying
	})

}

func getStd() {
	for _, s := range studs {
		stud := reflect.ValueOf(s)
		for i := 0; i < stud.NumField(); i++ {
			abs := math.Abs(stud.Field(i).Interface().(float64) - mean[i])
			std[i] += abs * abs
		}
	}
	for i := range std {
		std[i] /= float64(count)
		std[i] = math.Sqrt(std[i])
	}

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
			Arithmancy:                f[0],
			Astronomy:                 f[1],
			Herbology:                 f[2],
			DefenseAgainsttheDarkArts: f[3],
			Divination:                f[4],
			MuggleStudies:             f[5],
			AncientRunes:              f[6],
			HistoryofMagic:            f[7],
			Transfiguration:           f[8],
			Potions:                   f[9],
			CareofMagicalCreatures:    f[10],
			Charms:                    f[11],
			Flying:                    f[12],
		},
		)
		count++
	}
	// fmt.Println(len(studs), cap(studs))
	// get mean
	for i := range mean {
		mean[i] /= float64(count)
	}
	getStd()
	// getQuartiles()
	print()
}
