import { dbClient } from './common/constants/dbClient.js';
import { createEndpoints } from './common/functions/createEndpoints.js';
import { initialize } from './common/functions/initialize.js';
import { scrapeWebSchedules } from './common/functions/scrapeWebSchedules.js';

await dbClient.connect();
createEndpoints(initialize());
scrapeWebSchedules();