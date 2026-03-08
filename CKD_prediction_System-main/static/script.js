document.addEventListener('DOMContentLoaded', () => {
    
  
    const autoFillBtn = document.getElementById('autoFillBtn');
    if (autoFillBtn) {
        autoFillBtn.addEventListener('click', () => {
            const setVal = (id, val) => {
                const el = document.getElementById(id);
                if (el) el.value = val;
            };
            const setSelect = (id, val) => {
                const el = document.getElementById(id);
                if (el) el.value = val;
            };

            
            setVal('age', 58);
            setVal('bp', 90);
            setVal('hemo', 9.0);
            setVal('pcv', 28);
            setVal('wbcc', 9500);
            setVal('rbcc', 3.2);
            setVal('bgr', 160);
            setVal('bu', 55);
            setVal('sc', 3.8);
            setVal('sod', 130);
            setVal('pot', 5.2);
            setVal('sg', '1.010');
            setSelect('al', '3');
            setSelect('su', '2');
            setSelect('rbc', 'abnormal');
            setSelect('pc', 'abnormal');
            setSelect('pcc', 'present');
            setSelect('ba', 'present');
            setSelect('htn', 'yes');
            setSelect('dm', 'yes');
            setSelect('cad', 'no');
            setSelect('appet', 'poor');
            setSelect('pe', 'yes');
            setSelect('ane', 'yes');
        });
    }

  
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', () => {
            predictWithRandomForest();
        });
    }

    async function predictWithRandomForest() {
        const btn = document.getElementById('analyzeBtn');
        const originalText = btn.innerText;
        btn.innerText = "Analyzing...";
        btn.disabled = true;

       
        const parseNum = (id) => parseFloat(document.getElementById(id)?.value) || 0;
        const parseVal = (id) => document.getElementById(id)?.value || '';

       
        const inputData = {
            age: parseNum('age'),
            bp: parseNum('bp'),
            sg: parseFloat(parseVal('sg')),
            al: parseInt(parseVal('al')),
            su: parseInt(parseVal('su')),
            rbc: parseVal('rbc'),       
            pc: parseVal('pc'),         
            pcc: parseVal('pcc'),       
            ba: parseVal('ba'),         
            bgr: parseNum('bgr'),
            bu: parseNum('bu'),
            sc: parseNum('sc'),
            sod: parseNum('sod'),
            pot: parseNum('pot'),
            hemo: parseNum('hemo'),
            pcv: parseNum('pcv'),
            wbcc: parseNum('wbcc'),
            rbcc: parseNum('rbcc'),
            htn: parseVal('htn'),      
            dm: parseVal('dm'),         
            cad: parseVal('cad'),       
            appet: parseVal('appet'),  
            pe: parseVal('pe'),         
            ane: parseVal('ane')        
        };

        try {
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(inputData)
            });

            if (!response.ok) {
                throw new Error('Server error');
            }

            const result = await response.json();

           
            updateDashboardUI(result);

        } catch (error) {
            console.error('Prediction Error:', error);
            alert("Error connecting to prediction model. Ensure main.py is running.");
        } finally {
            btn.innerText = originalText;
            btn.disabled = false;
        }
    }

    function updateDashboardUI(data) {
        const resultBox = document.getElementById('resultBox');
        const placeholder = document.getElementById('placeholder');
        const predictionText = document.getElementById('predictionText');
        const riskBar = document.getElementById('riskBar');
        const riskPct = document.getElementById('riskPct');
        const factorList = document.getElementById('factorList');

        if (resultBox) {
            if (placeholder) placeholder.style.display = 'none';
            resultBox.style.display = 'block';

            const score = data.probability;
            
            
            let label = 'NOT CKD';
            let color = '#10b981'; 

            if (score > 40 && score <= 75) {
                label = 'NOT CKD';
                color = '#f59e0b'; 
            } else if (score > 75) {
                label = 'CKD Detected';
                color = '#ef4444'; 
            }

            predictionText.innerText = label;
            predictionText.style.color = color;
            
            riskBar.style.width = score + '%';
            riskBar.style.backgroundColor = color;
            riskPct.innerText = score + "% Probability";

          
            factorList.innerHTML = '';
            const note = document.createElement('li');
            note.innerText = `Model Confidence: ${data.probability}% (Random Forest)`;
            factorList.appendChild(note);
            
            const disclaimer = document.createElement('li');
            disclaimer.innerText = "Based on 24 medical parameters.";
            disclaimer.style.fontSize = "0.8em";
            disclaimer.style.color = "#94a3b8";
            factorList.appendChild(disclaimer);
        }
    }

  
    const contactForm = document.getElementById('contactForm');
    if (contactForm) {
        contactForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const successMsg = document.getElementById('successMsg');
            if (successMsg) {
                successMsg.style.display = 'block';
                contactForm.reset();
                setTimeout(() => {
                    successMsg.style.display = 'none';
                }, 3000);
            }
        });
    }

    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const navLinks = document.getElementById('navLinks');
    
    if (mobileMenuBtn && navLinks) {
        mobileMenuBtn.addEventListener('click', () => {
            navLinks.classList.toggle('active');
        });
    }
});