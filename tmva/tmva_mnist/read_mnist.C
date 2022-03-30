void read_mnist()
{
	TCanvas *c1 = new TCanvas();
	TFile *input = new TFile("mnist.root", "READ");

	TTree *tree = (TTree*)input->Get("train");

	Float_t x[784];
	Float_t y;
	
	tree->SetBranchAddress("x", &x);
	tree->SetBranchAddress("y", &y);

	int entries = tree->GetEntries();

	cout << entries << endl;

	//TH1F *hist = new TH1F("hist", "Histogram", 20, 0, 20);

	for(int i = 0; i < entries; i++) {
		tree->GetEntry(i);
		cout << x << " " << y << endl;
		//hist->Fill(y);
	}

	//hist->Draw();

	//input->Close();
}
