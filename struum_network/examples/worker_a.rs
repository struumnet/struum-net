use struum_network::worker::Worker;
use struum_types::network::*;
fn main(){
    let w_id = UID::new(0);
    let w1 = Worker{id:w_id};
    let _ = w1.listen();
}
